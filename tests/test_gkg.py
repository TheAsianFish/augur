"""GDELT's GKG files as a discovery index, and the searcher's fallback to it."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from augur.clock import FixedClock
from augur.errors import SearchError
from augur.ratelimit import RateLimiter
from augur.research.gdelt import GdeltSearcher, ResponseCache
from augur.research.gkg import (
    GkgIndex,
    batch_stamps,
    build_index,
    extract_rows,
    parse_master,
)
from augur.research.wayback import WaybackSearcher

from .test_gdelt import GDELT_HOST, NOW, FakeTime, FakeWeb

MASTER = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


def dt(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def gkg_line(stamp: str, url: str, title: str | None) -> str:
    fields = [""] * 27
    fields[0] = f"{stamp}-1"
    fields[1] = stamp
    fields[4] = url
    if title is not None:
        fields[26] = f"<PAGE_AUTHORS>x</PAGE_AUTHORS><PAGE_TITLE>{title}</PAGE_TITLE>"
    return "\t".join(fields)


def make_zip(stamp: str, lines: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(f"{stamp}.gkg.csv", "\n".join(lines) + "\n")
    return buf.getvalue()


def md5(b: bytes) -> str:
    return hashlib.md5(b, usedforsecurity=False).hexdigest()


# -- batch_stamps ------------------------------------------------------------------------


def test_batch_stamps_exclusive_start_inclusive_end_aligned() -> None:
    assert batch_stamps(dt(2025, 1, 1, 0, 7), dt(2025, 1, 1, 0, 45)) == [
        "20250101001500",
        "20250101003000",
        "20250101004500",
    ]


def test_batch_stamps_start_on_boundary_is_excluded() -> None:
    assert batch_stamps(dt(2025, 1, 1, 0, 15), dt(2025, 1, 1, 0, 30)) == ["20250101003000"]


def test_batch_stamps_end_before_next_batch_is_empty() -> None:
    assert batch_stamps(dt(2025, 1, 1, 0, 1), dt(2025, 1, 1, 0, 14)) == []


def test_batch_stamps_cross_midnight() -> None:
    assert batch_stamps(dt(2025, 1, 1, 23, 40), dt(2025, 1, 2, 0, 15)) == [
        "20250101234500",
        "20250102000000",
        "20250102001500",
    ]


# -- parse_master ------------------------------------------------------------------------


def test_parse_master_keeps_only_gkg_files_and_upgrades_to_https() -> None:
    text = "\n".join(
        [
            "100 AAAA http://data.gdeltproject.org/gdeltv2/20250101001500.export.CSV.zip",
            "200 BBBB http://data.gdeltproject.org/gdeltv2/20250101001500.gkg.csv.zip",
            "300 CCCC http://data.gdeltproject.org/gdeltv2/20250101001500.mentions.CSV.zip",
            "400 DDDD https://data.gdeltproject.org/gdeltv2/20250101003000.gkg.csv.zip",
            "garbage line",
            "",
        ]
    )
    found = parse_master(text)
    assert found == {
        "20250101001500": (
            200,
            "bbbb",
            "https://data.gdeltproject.org/gdeltv2/20250101001500.gkg.csv.zip",
        ),
        "20250101003000": (
            400,
            "dddd",
            "https://data.gdeltproject.org/gdeltv2/20250101003000.gkg.csv.zip",
        ),
    }


# -- extract_rows ------------------------------------------------------------------------


def test_extract_rows_keeps_only_titled_http_rows_of_this_batch() -> None:
    stamp = "20250101001500"
    lines = [
        gkg_line(stamp, "https://a.example.com/1", "  Bank   holds  rates "),
        gkg_line(stamp, "http://b.example.com/2", "Second"),
        gkg_line("20250101000000", "https://c.example.com/3", "Other batch"),
        gkg_line(stamp, "https://d.example.com/4", None),
        gkg_line(stamp, "ftp://e.example.com/5", "Not http"),
        "\t".join([""] * 4 + ["https://f.example.com/6"]),
        gkg_line(stamp, "https://g.example.com/7", "   "),
    ]
    assert extract_rows(stamp, make_zip(stamp, lines)) == [
        (stamp, "https://a.example.com/1", "Bank holds rates"),
        (stamp, "http://b.example.com/2", "Second"),
    ]


def test_extract_rows_rejects_a_zip_with_two_files() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a", "x")
        z.writestr("b", "y")
    with pytest.raises(SearchError, match="expected one file"):
        extract_rows("20250101001500", buf.getvalue())


# -- build_index -------------------------------------------------------------------------

S1, S2, S3 = "20250101001500", "20250101003000", "20250101004500"


async def test_build_index_retries_a_bad_md5_and_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slept: list[float] = []

    async def no_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("augur.research.gkg.asyncio.sleep", no_sleep)
    z1 = make_zip(S1, [gkg_line(S1, "https://a.example.com/1", "Alpha beat")])
    z2 = make_zip(
        S2,
        [
            gkg_line(S2, "https://b.example.com/2", "Beta  line"),
            gkg_line(S2, "https://c.example.com/3", "Gamma"),
        ],
    )
    base = "http://data.gdeltproject.org/gdeltv2/"
    master = (
        f"{len(z1)} {md5(z1)} {base}{S1}.gkg.csv.zip\n{len(z2)} {md5(z2)} {base}{S2}.gkg.csv.zip\n"
    )
    hits: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == MASTER:
            return httpx.Response(200, text=master)
        hits[url] = hits.get(url, 0) + 1
        if url.endswith(f"{S1}.gkg.csv.zip"):
            return httpx.Response(200, content=z1)
        if hits[url] == 1:
            return httpx.Response(200, content=z2 + b"corrupt")
        return httpx.Response(200, content=z2)

    out = tmp_path / "sub" / "index.gz"
    messages: list[str] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        report = await build_index(
            http, dt(2025, 1, 1, 0, 0), dt(2025, 1, 1, 0, 45), out, progress=messages.append
        )
    assert report.files == 2
    assert report.missing == [S3]
    assert report.articles == 3
    assert hits[f"https://data.gdeltproject.org/gdeltv2/{S2}.gkg.csv.zip"] == 2
    assert slept == [5.0]
    assert messages == ["gkg: 2/2 files"]
    with gzip.open(out, "rt", encoding="utf-8") as fh:
        text = fh.read()
    assert text.splitlines() == [
        f"{S1}\thttps://a.example.com/1\tAlpha beat",
        f"{S2}\thttps://b.example.com/2\tBeta line",
        f"{S2}\thttps://c.example.com/3\tGamma",
    ]
    assert report.sha256 == hashlib.sha256(text.encode()).hexdigest()
    manifest = report.manifest(dt(2025, 1, 1, 0, 0), dt(2025, 1, 1, 0, 45))
    assert manifest["missing_from_master"] == [S3]
    assert manifest["articles"] == 3


async def test_build_index_gives_up_after_four_bad_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("augur.research.gkg.asyncio.sleep", no_sleep)
    z1 = make_zip(S1, [gkg_line(S1, "https://a.example.com/1", "Alpha")])
    master = f"{len(z1)} {md5(z1)} http://data.gdeltproject.org/gdeltv2/{S1}.gkg.csv.zip\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == MASTER:
            return httpx.Response(200, text=master)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(SearchError, match="no good download after 4 attempts"):
            await build_index(http, dt(2025, 1, 1, 0, 0), dt(2025, 1, 1, 0, 15), tmp_path / "i.gz")


async def test_build_index_refuses_when_the_master_list_is_unavailable(tmp_path: Path) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(500))
    ) as http:
        with pytest.raises(SearchError, match="master list"):
            await build_index(http, dt(2025, 1, 1, 0, 0), dt(2025, 1, 1, 0, 15), tmp_path / "i.gz")


# -- GkgIndex ----------------------------------------------------------------------------

VOCAB = {"central", "bank", "rates", "december", "inflation", "steady"}


def rows(*items: tuple[str, str, str]) -> list[tuple[str, str, str]]:
    return list(items)


def urls(body: str) -> list[str]:
    return [a["url"] for a in json.loads(body)["articles"]]


WIDE = (dt(2025, 1, 1), dt(2025, 1, 2))


def test_index_needs_three_title_words() -> None:
    index = GkgIndex(
        rows(
            (S1, "https://a/1", "Central bank holds rates"),
            (S1, "https://a/2", "Central bank news"),
        ),
        VOCAB,
    )
    assert urls(index.artlist(["central", "bank", "rates", "december"], *WIDE)) == ["https://a/1"]


def test_index_with_fewer_keywords_needs_all_of_them() -> None:
    index = GkgIndex(
        rows((S1, "https://a/1", "Central bank"), (S1, "https://a/2", "Central rates")), VOCAB
    )
    assert urls(index.artlist(["central", "bank"], *WIDE)) == ["https://a/1"]
    assert index.artlist([], *WIDE) == json.dumps({"articles": []})


def test_index_ignores_numeric_keywords() -> None:
    index = GkgIndex(rows((S1, "https://a/1", "Central bank rates")), VOCAB | {"2025"})
    assert urls(index.artlist(["central", "bank", "rates", "2025", "2026"], *WIDE)) == [
        "https://a/1"
    ]


def test_index_orders_by_matches_then_newest() -> None:
    index = GkgIndex(
        rows(
            (S1, "https://a/old3", "Central bank rates"),
            (S2, "https://a/new3", "Central bank rates"),
            (S1, "https://a/old4", "Central bank rates december"),
            (S3, "https://a/new3b", "Rates bank central"),
        ),
        VOCAB,
    )
    got = urls(index.artlist(["central", "bank", "rates", "december"], *WIDE))
    assert got == ["https://a/old4", "https://a/new3b", "https://a/new3", "https://a/old3"]


def test_index_window_start_exclusive_end_inclusive() -> None:
    index = GkgIndex(
        rows(
            (S1, "https://a/at-start", "Central bank rates"),
            (S2, "https://a/inside", "Central bank rates"),
            (S3, "https://a/at-end", "Central bank rates"),
            ("20250101010000", "https://a/after", "Central bank rates"),
        ),
        VOCAB,
    )
    start, end = dt(2025, 1, 1, 0, 15), dt(2025, 1, 1, 0, 45)
    got = urls(index.artlist(["central", "bank", "rates"], start, end))
    assert got == ["https://a/at-end", "https://a/inside"]


def test_index_answer_has_doc_shape_and_caps_results() -> None:
    many = [(S1, f"https://news.example.com/{i}", "Central bank rates") for i in range(15)]
    body = GkgIndex(many, VOCAB).artlist(["central", "bank", "rates"], *WIDE)
    articles = json.loads(body)["articles"]
    assert len(articles) == 10
    first = articles[0]
    assert first["seendate"] == "20250101T001500Z"
    assert first["domain"] == "news.example.com"
    assert first["route"] == "gkg"
    assert first["title"] == "Central bank rates"


def test_index_only_vocabulary_words_are_indexed() -> None:
    index = GkgIndex(
        rows(
            (S1, "https://a/1", "Central bank rates"),
            (S1, "https://a/2", "Unrelated football match"),
        ),
        {"central", "bank"},
    )
    assert len(index._rows) == 1  # the second title holds no vocabulary word
    assert set(index._postings) == {"central", "bank"}
    # "rates" is outside the vocabulary, so it can never count towards a match.
    assert index.artlist(["central", "bank", "rates"], *WIDE) == json.dumps({"articles": []})


def test_load_keeps_the_first_batch_of_a_duplicate_url(tmp_path: Path) -> None:
    path = tmp_path / "i.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(f"{S1}\thttps://a/dup\tCentral bank rates\n")
        fh.write(f"{S3}\thttps://a/dup\tCentral bank rates\n")
        fh.write("bad line\n")
        fh.write(f"{S2}\thttps://a/other\tCentral bank rates\n")
    index = GkgIndex.load(path, {"Central", "bank", "rates"})
    # A cutoff between the two batches still finds the article, at its first batch.
    got = urls(index.artlist(["central", "bank", "rates"], dt(2025, 1, 1), dt(2025, 1, 1, 0, 20)))
    assert got == ["https://a/dup"]
    assert urls(
        index.artlist(["central", "bank", "rates"], dt(2025, 1, 1, 0, 20), dt(2025, 1, 2))
    ) == ["https://a/other"]


# -- the searcher's fallback -------------------------------------------------------------

AS_OF = dt(2025, 1, 1, 12, 0)
GOOD = "https://news.example.com/story"
LATE = "https://late.example.com/story"
QUERY = "central bank rates December"
CAPTURE = "20241230093000"


class CountingIndex(GkgIndex):
    def __init__(self, rows: list[tuple[str, str, str]], vocab: set[str]) -> None:
        super().__init__(rows, vocab)
        self.calls = 0

    def artlist(self, keywords, start, end):  # type: ignore[no-untyped-def]  # mirrors the base signature
        self.calls += 1
        return super().artlist(keywords, start, end)


def refused_searcher(
    index: GkgIndex,
    cache: ResponseCache | None = None,
    captures: dict[str, list[str]] | None = None,
) -> tuple[GdeltSearcher, httpx.AsyncClient, FakeWeb]:
    web = FakeWeb(
        captures=captures if captures is not None else {GOOD: [CAPTURE], LATE: [CAPTURE]},
        gdelt_responses=[httpx.Response(429)],
    )
    t = FakeTime()
    http = httpx.AsyncClient(transport=httpx.MockTransport(web))
    clock = FixedClock(NOW)
    wayback = WaybackSearcher(
        http, clock, limiter=RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    )
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    searcher = GdeltSearcher(
        http, clock, wayback, cache=cache, limiter=limiter, backoff=(0.0,), index=index
    )
    return searcher, http, web


def fallback_index() -> CountingIndex:
    return CountingIndex(
        rows(
            ("20241220080000", GOOD, "Central bank holds rates steady in December"),
            ("20250101123000", LATE, "Central bank rates December outlook"),  # after the cutoff
        ),
        VOCAB,
    )


async def test_api_refusal_falls_back_to_the_index_and_stays_there(tmp_path: Path) -> None:
    index = fallback_index()
    searcher, http, web = refused_searcher(index, ResponseCache(tmp_path / "c"))
    async with http:
        first = await searcher.search(QUERY, AS_OF, 5)
        assert searcher.api_refused
        api_calls = len(web.gdelt_requests())
        assert api_calls == 2  # one try and one retry, both 429
        second = await searcher.search("central bank rates steady", AS_OF, 5)
    assert [s.url.rsplit("/", 3)[-1] for s in first] == ["story"]
    assert first[0].url == f"https://web.archive.org/web/{CAPTURE}/{GOOD}"
    assert len(second) == 1
    assert len(web.gdelt_requests()) == api_calls  # the API was not asked again
    assert index.calls == 2


async def test_index_article_after_the_cutoff_is_never_returned_or_looked_up(
    tmp_path: Path,
) -> None:
    searcher, http, web = refused_searcher(fallback_index(), ResponseCache(tmp_path / "c"))
    async with http:
        sources = await searcher.search(QUERY, AS_OF, 5)
    assert all(LATE not in s.url for s in sources)
    asked = [str(r.url) for r in web.wayback_requests()]
    assert asked  # the good article was looked up
    assert not any("late.example.com" in u for u in asked)
    assert web.other_hosts() == []


async def test_index_answer_is_cached_under_a_gkg_key(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c")
    index = fallback_index()
    searcher, http, _ = refused_searcher(index, cache)
    async with http:
        await searcher.search(QUERY, AS_OF, 5)
        assert index.calls == 1
        await searcher.search(QUERY, AS_OF, 5)
    assert index.calls == 1  # the repeat came from the cache
    files = list((tmp_path / "c").glob("*.json"))
    keys = [json.loads(f.read_text())["url"] for f in files]
    assert any(k.startswith("gkg:") for k in keys)


async def test_a_working_api_never_touches_the_index() -> None:
    index = fallback_index()
    web = FakeWeb([], {})
    t = FakeTime()
    http = httpx.AsyncClient(transport=httpx.MockTransport(web))
    clock = FixedClock(NOW)
    wayback = WaybackSearcher(http, clock)
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    searcher = GdeltSearcher(http, clock, wayback, limiter=limiter, index=index)
    async with http:
        assert await searcher.search(QUERY, AS_OF, 5) == []
    assert not searcher.api_refused
    assert index.calls == 0
    assert GDELT_HOST in {r.url.host for r in web.requests}


async def test_without_an_index_the_refusal_still_raises() -> None:
    web = FakeWeb(gdelt_responses=[httpx.Response(429)])
    t = FakeTime()
    http = httpx.AsyncClient(transport=httpx.MockTransport(web))
    clock = FixedClock(NOW)
    limiter = RateLimiter(1000.0, burst=100, monotonic=t.monotonic, sleep=t.sleep)
    searcher = GdeltSearcher(
        http, clock, WaybackSearcher(http, clock), limiter=limiter, backoff=(0.0,)
    )
    async with http:
        with pytest.raises(SearchError):
            await searcher.search(QUERY, AS_OF, 5)
    assert not searcher.api_refused
