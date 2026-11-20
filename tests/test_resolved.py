from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from augur import resolved
from augur.errors import AugurError
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from tests.fakes.metaculus import FAKE_TOKEN, FakeMetaculus, binary_post, group_post

BASE = "https://www.metaculus.com/api"
RELEASED = date(2026, 6, 4)
SLUG = "minibench-test"


def rpost(
    post_id: int,
    *,
    resolution: str | None = "yes",
    opened: str = "2026-07-01T00:00:00Z",
    resolved_at: str | None = "2026-08-01T00:00:00Z",
    qtype: str = "binary",
    status: str = "resolved",
) -> dict[str, Any]:
    post = binary_post(post_id, post_id + 10_000, status=status)
    q = post["question"]
    q.update(
        resolution=resolution,
        open_time=opened,
        actual_resolve_time=resolved_at,
        scheduled_close_time="2026-07-30T00:00:00Z",
        type=qtype,
    )
    return post


async def _no_sleep(_: float) -> None:
    return None


@pytest.fixture
def fake() -> FakeMetaculus:
    return FakeMetaculus(token=FAKE_TOKEN)


@pytest.fixture
async def client(fake: FakeMetaculus) -> AsyncIterator[MetaculusClient]:
    limiter = RateLimiter(rate_per_second=1000.0, burst=1000, max_concurrent=4, sleep=_no_sleep)
    async with httpx.AsyncClient(transport=fake.transport) as http:
        yield MetaculusClient(BASE, SecretStr(FAKE_TOKEN), http, limiter, page_size=3)


async def load(client: MetaculusClient, **kw: Any) -> resolved.ResolvedLoadReport:
    return await resolved.load_tournament(client, SLUG, released=RELEASED, **kw)


async def test_keeps_a_good_question(fake: FakeMetaculus, client: MetaculusClient) -> None:
    fake.add_post(rpost(1, resolution="no"), SLUG)
    report = await load(client)
    assert report.total == 1
    assert report.removed == {}
    (item,) = report.kept
    assert item.outcome == 0
    assert item.resolution == "no"
    assert item.resolved_at == datetime(2026, 8, 1, tzinfo=UTC)
    assert item.question.post_id == 1
    assert item.tournament == SLUG
    assert item.question.my_latest_forecast is None
    assert report.lines()[-1].strip() == "kept       1"


async def test_each_filter_counts_its_reason(fake: FakeMetaculus, client: MetaculusClient) -> None:
    fake.add_post(rpost(1), SLUG)  # kept
    group = group_post(2)
    group["status"] = "resolved"
    fake.add_post(group, SLUG)
    fake.add_post(rpost(3, qtype="numeric"), SLUG)
    fake.add_post(rpost(4, resolution="annulled"), SLUG)
    fake.add_post(rpost(5, resolution="ambiguous"), SLUG)
    fake.add_post(rpost(6, resolved_at=None), SLUG)
    fake.add_post(rpost(7, resolution=None), SLUG)
    fake.add_post(rpost(8, opened="2026-07-10T00:00:00Z", resolved_at="2026-07-01T00:00:00Z"), SLUG)
    fake.add_post(rpost(9, opened="2026-05-01T00:00:00Z"), SLUG)
    fake.add_post(rpost(10, resolution="maybe"), SLUG)
    fake.add_post(rpost(11, status="open"), SLUG)  # never listed as resolved
    fake.add_post(rpost(12), "other-tournament")
    broken = rpost(13)
    broken["question"]["open_time"] = None
    broken["open_time"] = None
    fake.add_post(broken, SLUG)

    report = await load(client)
    assert [k.question.post_id for k in report.kept] == [1]
    assert report.total == 11
    assert report.removed == {
        resolved.R_GROUP: 1,
        resolved.R_TYPE: 1,
        resolved.R_ANNULLED: 2,
        resolved.R_NOT_RESOLVED: 1,
        resolved.R_HIDDEN: 1,
        resolved.R_BEFORE_OPEN: 1,
        resolved.R_RELEASE: 1,
        resolved.R_UNPARSED: 1,
        resolved.R_NO_OPEN: 1,
    }
    assert sum(report.removed.values()) + len(report.kept) == report.total
    assert any("annulled or ambiguous" in line for line in report.lines())


async def test_question_types_widen_the_type_filter(
    fake: FakeMetaculus, client: MetaculusClient
) -> None:
    fake.add_post(rpost(1, qtype="numeric", resolution="12.5"), SLUG)
    report = await load(client, question_types=("binary", "numeric"))
    # numeric outcomes are not parseable as 0 or 1
    assert report.removed == {resolved.R_UNPARSED: 1}


async def test_release_boundary(fake: FakeMetaculus, client: MetaculusClient) -> None:
    fake.add_post(rpost(1, opened="2026-06-04T23:59:00Z"), SLUG)  # the release day
    fake.add_post(rpost(2, opened="2026-06-05T00:00:00Z"), SLUG)  # the day after
    report = await load(client)
    assert [k.question.post_id for k in report.kept] == [2]
    assert report.removed == {resolved.R_RELEASE: 1}


async def test_pagination_and_limit(fake: FakeMetaculus, client: MetaculusClient) -> None:
    for i in range(1, 9):
        fake.add_post(rpost(i), SLUG)
    report = await load(client)
    assert report.total == 8
    assert len(report.kept) == 8
    assert sum(1 for r in fake.requests if r.url.path.endswith("/posts/")) == 3
    assert all(r.url.params["statuses"] == "resolved" for r in fake.requests)
    limited = await load(client, limit=5)
    assert [k.question.post_id for k in limited.kept] == [1, 2, 3, 4, 5]
    assert limited.removed == {"over the limit": 3}


async def test_numeric_tournament_id(fake: FakeMetaculus, client: MetaculusClient) -> None:
    fake.add_post(rpost(1), "33022")
    report = await resolved.load_tournament(client, 33022, released=RELEASED)
    assert report.kept[0].tournament_id == 33022


async def test_snapshot_round_trip(
    fake: FakeMetaculus, client: MetaculusClient, tmp_path: Path
) -> None:
    fake.add_post(rpost(1, opened="2026-06-04T12:00:00Z"), SLUG)
    fake.add_post(rpost(2, opened="2026-07-01T00:00:00Z", resolution="no"), SLUG)
    fake.add_post(rpost(3, opened="2026-09-01T00:00:00Z", resolved_at="2026-10-01T00:00:00Z"), SLUG)
    report = await resolved.load_tournament(client, SLUG, released=date(2026, 1, 1))
    path = tmp_path / "sub" / "snap.json"
    resolved.write_snapshot(report, path)
    text = path.read_text(encoding="utf-8")
    assert FAKE_TOKEN not in text
    assert "my_latest_forecast" not in text

    same = resolved.load_snapshot(path, date(2026, 1, 1))
    assert same.kept == report.kept
    assert same.removed == {}

    later = resolved.load_snapshot(path, RELEASED)
    assert [k.question.post_id for k in later.kept] == [2, 3]
    assert later.removed == {resolved.R_RELEASE: 1}
    assert later.total == 3

    again = tmp_path / "again.json"
    resolved.write_snapshot(same, again)
    assert again.read_text(encoding="utf-8") == text


def test_bad_snapshot_names_the_next_step(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(AugurError, match="regenerate"):
        resolved.load_snapshot(path, RELEASED)
