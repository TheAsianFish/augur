"""GDELT's 15-minute GKG files as a local discovery index (M13, registration item 7).

GDELT's DOC 2.0 API searches the same crawl, but answers some addresses with
HTTP 429 on every request. Its static files do not rate-limit, so this module
builds the part of the index a backtest needs from them:

* :func:`build_index` downloads every GKG 2.1 file whose 15-minute batch stamp
  is after ``start`` and at or before ``end``, checks each against the MD5 in
  GDELT's master file list, and keeps one line per article: batch stamp, URL,
  page title (from the ``<PAGE_TITLE>`` element of the extras field);
* :class:`GkgIndex` answers a query with the articles whose title holds
  enough of its keywords, in the shape of a DOC ``artlist`` answer, so the
  searcher treats both routes alike.

The batch stamp is when GDELT processed the article: the same instant the
API calls ``seendate``, and checked against the cutoff the same way. A title
is discovery only: the text a model reads still comes from a Wayback capture
at or before the cutoff.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from augur.clock import ensure_utc
from augur.errors import SearchError
from augur.log import get_logger

MASTER_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
STEP = timedelta(minutes=15)
MIN_MATCH = 3
MAX_RESULTS = 10
_TITLE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)
_WORD = re.compile(r"[a-z0-9]+")

log = get_logger("augur.research.gkg")


def batch_stamps(start: datetime, end: datetime) -> list[str]:
    """Every 15-minute batch stamp after ``start`` and at or before ``end``."""
    start, end = ensure_utc(start), ensure_utc(end)
    first = start.replace(minute=start.minute - start.minute % 15, second=0, microsecond=0)
    t = first if first > start else first + STEP
    out: list[str] = []
    while t <= end:
        out.append(t.strftime("%Y%m%d%H%M%S"))
        t += STEP
    return out


def parse_master(text: str) -> dict[str, tuple[int, str, str]]:
    """GKG files in GDELT's master list: stamp -> (size, md5, url)."""
    found: dict[str, tuple[int, str, str]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[2].endswith(".gkg.csv.zip"):
            continue
        stamp = parts[2].rsplit("/", 1)[-1].split(".", 1)[0]
        if len(stamp) == 14 and stamp.isdigit() and parts[0].isdigit():
            # The list names plain-http URLs, which the host redirects to https.
            url = "https://" + parts[2].split("://", 1)[-1]
            found[stamp] = (int(parts[0]), parts[1].lower(), url)
    return found


def extract_rows(stamp: str, payload: bytes) -> list[tuple[str, str, str]]:
    """``(stamp, url, title)`` for each article of one zipped GKG file with a title."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise SearchError(f"GKG {stamp}: expected one file in the zip, found {len(names)}")
        text = archive.read(names[0]).decode("utf-8", errors="replace")
    rows: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) < 27 or cols[1] != stamp:
            continue
        url = cols[4].strip()
        match = _TITLE.search(cols[26])
        if not url.lower().startswith(("http://", "https://")) or match is None:
            continue
        title = " ".join(match.group(1).split())
        if title:
            rows.append((stamp, url, title))
    return rows


@dataclass(frozen=True)
class IndexReport:
    files: int
    missing: list[str]
    articles: int
    sha256: str
    """SHA-256 of the written index, for the manifest."""

    def manifest(self, start: datetime, end: datetime) -> dict[str, Any]:
        return {
            "source": MASTER_URL,
            "start_exclusive": ensure_utc(start).isoformat(),
            "end_inclusive": ensure_utc(end).isoformat(),
            "files": self.files,
            "missing_from_master": self.missing,
            "articles": self.articles,
            "index_sha256": self.sha256,
        }


async def build_index(
    http: httpx.AsyncClient,
    start: datetime,
    end: datetime,
    out: Path,
    *,
    concurrency: int = 6,
    master_url: str = MASTER_URL,
    progress: Callable[[str], None] = print,
) -> IndexReport:
    """Download and reduce every GKG file in ``(start, end]`` into ``out`` (gzipped TSV)."""
    master = await http.get(master_url, timeout=httpx.Timeout(300.0, connect=10.0))
    if master.status_code != 200:
        raise SearchError(f"GDELT master list: HTTP {master.status_code}; retry later")
    listed = parse_master(master.text)
    stamps = batch_stamps(start, end)
    missing = [s for s in stamps if s not in listed]
    wanted = [s for s in stamps if s in listed]
    results: dict[str, list[tuple[str, str, str]]] = {}
    slots = asyncio.Semaphore(concurrency)

    async def one(stamp: str) -> None:
        size, md5, url = listed[stamp]
        async with slots:
            for attempt in range(4):
                try:
                    response = await http.get(url, timeout=httpx.Timeout(120.0, connect=10.0))
                except httpx.HTTPError as exc:
                    log.warning("gkg_transport_error", stamp=stamp, error=str(exc))
                    await asyncio.sleep(5.0 * (attempt + 1))
                    continue
                body = response.content
                if (
                    response.status_code == 200
                    and hashlib.md5(body, usedforsecurity=False).hexdigest() == md5
                ):
                    if len(body) != size:
                        raise SearchError(f"GKG {stamp}: size {len(body)} != listed {size}")
                    results[stamp] = extract_rows(stamp, body)
                    return
                log.warning("gkg_bad_download", stamp=stamp, status=response.status_code)
                await asyncio.sleep(5.0 * (attempt + 1))
        raise SearchError(f"GKG {stamp}: no good download after 4 attempts; re-run to resume")

    # Bounded: one task per file, at most ``concurrency`` downloading at once.
    for i in range(0, len(wanted), 96):
        chunk = wanted[i : i + 96]
        await asyncio.gather(*(one(s) for s in chunk))
        progress(f"gkg: {min(i + 96, len(wanted))}/{len(wanted)} files")
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    articles = 0
    with gzip.open(out, "wt", encoding="utf-8", compresslevel=6) as fh:
        for stamp in wanted:
            for row in results[stamp]:
                line = "\t".join(c.replace("\t", " ") for c in row) + "\n"
                digest.update(line.encode("utf-8"))
                fh.write(line)
                articles += 1
    return IndexReport(len(wanted), missing, articles, digest.hexdigest())


def _stamp_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


class GkgIndex:
    """Articles by title words, from an index :func:`build_index` wrote.

    Only the words in ``vocabulary`` (every keyword the run's queries can
    ask for) are indexed, so a month of articles fits in memory.
    """

    def __init__(self, rows: Sequence[tuple[str, str, str]], vocabulary: set[str]) -> None:
        self._rows: list[tuple[str, str, str]] = []
        self._postings: dict[str, list[int]] = {}
        for row in rows:
            words = set(_WORD.findall(row[2].lower())) & vocabulary
            if not words:
                continue
            for w in words:
                self._postings.setdefault(w, []).append(len(self._rows))
            self._rows.append(row)

    @classmethod
    def load(cls, path: Path, vocabulary: set[str]) -> GkgIndex:
        def rows() -> Iterator[tuple[str, str, str]]:
            seen: set[str] = set()
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) != 3 or parts[1] in seen:
                        continue  # an article is kept at its first (earliest) batch
                    seen.add(parts[1])
                    yield parts[0], parts[1], parts[2]

        return cls(list(rows()), {w.lower() for w in vocabulary})

    def artlist(self, keywords: Sequence[str], start: datetime, end: datetime) -> str:
        """A DOC-style ``artlist`` answer for ``keywords`` over batches in ``(start, end]``.

        An article matches when its title holds at least :data:`MIN_MATCH` of
        the non-numeric keywords (all of them if there are fewer); matches
        are ordered by keywords matched, then newest first, then index order.
        """
        words = sorted({w.lower() for w in keywords if w and not w.isdigit()})
        need = min(MIN_MATCH, len(words))
        counts: dict[int, int] = {}
        for w in words:
            for i in self._postings.get(w, []):
                counts[i] = counts.get(i, 0) + 1
        lo, hi = ensure_utc(start), ensure_utc(end)
        hits = [
            (score, i)
            for i, score in counts.items()
            if need and score >= need and lo < _stamp_dt(self._rows[i][0]) <= hi
        ]
        hits.sort(key=lambda h: (-h[0], -int(self._rows[h[1]][0]), h[1]))
        articles = []
        for _score, i in hits[:MAX_RESULTS]:
            stamp, url, title = self._rows[i]
            articles.append(
                {
                    "url": url,
                    "title": title,
                    "seendate": stamp[:8] + "T" + stamp[8:] + "Z",
                    "domain": urlsplit(url).hostname or "",
                    "route": "gkg",
                }
            )
        return json.dumps({"articles": articles}, sort_keys=True)
