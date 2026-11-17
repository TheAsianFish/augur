"""``augur forecast`` end to end against the fake Metaculus server."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from augur.clock import FixedClock
from augur.config import Settings
from augur.errors import AugurError
from augur.leakage import RunMode, scan_ledger
from augur.ledger import Ledger
from augur.log import configure_logging
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from augur.runner import SAMPLES_DIR, ForecastRun, RunReport

from .conftest import T0
from .fakes.llm import ScriptedProvider
from .fakes.metaculus import FAKE_TOKEN, MINIBENCH_NUMERIC, FakeMetaculus, binary_post
from .fakes.search import FixedSearcher


def _fast_limiter() -> RateLimiter:
    return RateLimiter(1000, burst=1000, max_concurrent=4)


@pytest.fixture
def fake() -> FakeMetaculus:
    server = FakeMetaculus(clock=FixedClock(T0))
    server.add_post(binary_post(500, 501, title="Will the fake event happen?"), "minibench")
    server.add_post(binary_post(510, 511, title="A closed one", status="closed"), "minibench")
    server.add_fixture(MINIBENCH_NUMERIC, "minibench")
    return server


@pytest.fixture
async def http(fake: FakeMetaculus) -> AsyncIterator[httpx.AsyncClient]:
    async with fake.client() as client:
        yield client


def _settings(tmp_path: Path, **extra: object) -> Settings:
    return Settings.model_validate(
        {"provider": "replay", "model": "qwen2.5:7b", "ledger_path": tmp_path / "l.sqlite"} | extra
    )


async def _run(
    tmp_path: Path,
    *,
    client: MetaculusClient | None,
    provider: ScriptedProvider | None = None,
    submit: bool = False,
    samples: Path | None = None,
    max_questions: int = 10,
    out: io.StringIO | None = None,
    searcher: FixedSearcher | None = None,
    **settings: object,
) -> tuple[RunReport, Ledger]:
    clock = FixedClock(T0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger.open(tmp_path / "l.sqlite", clock)
    run = ForecastRun(
        settings=_settings(tmp_path, **settings),
        clock=clock,
        ledger=ledger,
        provider=provider or ScriptedProvider(),
        searchers=[searcher or FixedSearcher()],
        client=client,
        out=out or io.StringIO(),
    )
    report = await run.run("minibench", submit=submit, samples=samples, max_questions=max_questions)
    return report, ledger


def _client(http: httpx.AsyncClient, token: str | None = FAKE_TOKEN) -> MetaculusClient:
    return MetaculusClient(
        "https://www.metaculus.com/api",
        SecretStr(token) if token else None,
        http,
        _fast_limiter(),
    )


async def test_dry_run_on_samples_ledgers_everything_and_prints_the_comment(
    tmp_path: Path,
) -> None:
    out = io.StringIO()
    report, ledger = await _run(tmp_path, client=None, out=out)
    assert report.mode is RunMode.DRY_RUN
    assert report.source.startswith("samples:")
    n_samples = len(list(SAMPLES_DIR.glob("post_*.json")))
    assert report.count("forecast") == n_samples
    [first] = [r for r in report.reports if r.status == "forecast"][:1]
    assert first.probability == pytest.approx(0.40)
    run = report.run_id
    assert ledger.count("questions", run) == n_samples
    # planner + brief + five forecasters per question
    assert ledger.count("llm_calls", run) == 7 * n_samples
    assert ledger.count("forecasts", run) == 6 * n_samples
    assert ledger.count("submissions", run) == 2 * n_samples
    conn = ledger.connection
    statuses = {r[0] for r in conn.execute("SELECT status FROM submissions")}
    assert statuses == {"dry_run"}
    # The late source was stored, judged and rejected, and no call cites it.
    rejected = conn.execute(
        "SELECT count(*) FROM source_uses u JOIN sources s USING (source_id) "
        "WHERE u.admitted = 0 AND s.url = 'https://example.org/the-answer'"
    ).fetchone()[0]
    assert rejected == n_samples
    cited_late = conn.execute(
        "SELECT count(*) FROM call_sources JOIN sources USING (source_id) "
        "WHERE url = 'https://example.org/the-answer'"
    ).fetchone()[0]
    assert cited_late == 0
    text = out.getvalue()
    assert "would POST /api/questions/forecast/" in text
    assert '"probability_yes": 0.4' in text
    assert "| **Forecast: 40%**, the median of 5 independent forecasts" in text
    assert "It resolved Yes" not in text
    ledger.close()
    assert scan_ledger(tmp_path / "l.sqlite") == []


async def test_live_run_submits_once_with_a_private_comment_and_skips_next_time(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    out = io.StringIO()
    report, ledger = await _run(tmp_path, client=_client(http), submit=True, out=out)
    assert report.mode is RunMode.LIVE
    assert "posted this private comment:\n   | **Forecast: 40%**" in out.getvalue()
    forecast = [r for r in report.reports if r.status == "forecast"]
    assert [r.question.question_id for r in forecast] == [501]
    assert forecast[0].submitted
    reasons = {r.question.question_id: r.reason for r in report.reports if r.status == "skipped"}
    assert "not forecast yet" in reasons[46876]
    assert len(fake.forecasts) == 1
    sent = fake.forecasts[0]
    assert sent == {
        "question": 501,
        "source": "api",
        "probability_yes": 0.4,
        "probability_yes_per_category": None,
        "continuous_cdf": None,
    }
    assert len(fake.comments) == 1
    comment = fake.comments[0]
    assert comment["is_private"] is True
    assert comment["on_post"] == 500
    assert comment["text"].startswith("**Forecast: 40%**")
    assert "Base rate is low" in comment["text"]
    rows = ledger.connection.execute(
        "SELECT kind, status, http_status FROM submissions ORDER BY kind"
    ).fetchall()
    assert rows == [("comment", "submitted", 201), ("forecast", "submitted", 201)]
    ledger.close()

    second, ledger2 = await _run(tmp_path, client=_client(http), submit=True)
    assert second.count("forecast") == 0
    skipped = {r.question.question_id: r.reason for r in second.reports}
    assert skipped[501] == "already forecast by this account"
    assert len(fake.forecasts) == 1, "a second run must never forecast the same question"
    assert len(fake.comments) == 1
    ledger2.close()


async def test_with_a_token_but_no_submit_the_listing_is_live_and_nothing_is_sent(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    report, ledger = await _run(tmp_path, client=_client(http))
    assert report.source == "metaculus:minibench"
    assert report.count("forecast") == 1
    assert fake.forecasts == []
    assert fake.comments == []
    ledger.close()


async def test_submit_without_a_token_is_refused(tmp_path: Path, http: httpx.AsyncClient) -> None:
    with pytest.raises(AugurError, match="METACULUS_TOKEN"):
        await _run(tmp_path, client=_client(http, token=None), submit=True)


async def test_failed_draws_degrade_the_question_without_aborting_the_run(
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider([None, 20, None, 30, 40])
    report, ledger = await _run(tmp_path, client=None, provider=provider)
    first = report.reports[0]
    assert first.status == "forecast"
    assert first.probability == pytest.approx(0.30)
    assert "2 of 5 draws failed" in first.comment
    errors = ledger.connection.execute(
        "SELECT count(*) FROM llm_calls WHERE error IS NOT NULL"
    ).fetchone()[0]
    assert errors >= 2
    ledger.close()

    too_few = ScriptedProvider([None, None, None, 30, 40])
    report2, ledger2 = await _run(tmp_path / "b", client=None, provider=too_few)
    assert report2.reports[0].status == "failed"
    assert "only 2 of 5 forecasters answered" in report2.reports[0].reason
    ledger2.close()


async def test_a_failing_searcher_degrades_research_only(tmp_path: Path) -> None:
    report, ledger = await _run(tmp_path, client=None, searcher=FixedSearcher(fail=True))
    assert report.reports[0].status == "forecast"
    assert "No sources dated before the cutoff" in report.reports[0].comment
    ledger.close()


async def test_the_question_cap_and_the_cost_cap(
    tmp_path: Path, fake: FakeMetaculus, http: httpx.AsyncClient
) -> None:
    fake.add_post(binary_post(520, 521, title="Another open one"), "minibench")
    report, ledger = await _run(tmp_path, client=_client(http), max_questions=1)
    assert report.count("forecast") == 1
    assert any("over the cap of 1" in r.reason for r in report.reports)
    ledger.close()

    costly = ScriptedProvider(cost_per_call=0.2)
    report2, ledger2 = await _run(
        tmp_path / "b", client=_client(http), provider=costly, max_cost_per_run=0.5
    )
    assert any("run budget" in r.reason for r in report2.reports)
    assert report2.cost_usd <= 0.5 + 0.2
    ledger2.close()


async def test_no_token_reaches_the_logs_during_a_live_run(
    tmp_path: Path,
    fake: FakeMetaculus,
    http: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("METACULUS_TOKEN", FAKE_TOKEN)
    stream = io.StringIO()
    configure_logging("DEBUG", stream=stream)
    out = io.StringIO()
    fake.rate_limited = 1
    fake.retry_after = "0.01"
    await _run(tmp_path, client=_client(http), submit=True, out=out)
    logs = stream.getvalue()
    assert logs, "the run should have logged"
    assert FAKE_TOKEN not in logs
    assert FAKE_TOKEN not in out.getvalue()
    events = [json.loads(line) for line in logs.splitlines()]
    assert all("run_id" in e for e in events if e["event"] != "asknews skipped")
    assert any(e.get("question_id") == 501 for e in events)
    ledger_bytes = (tmp_path / "l.sqlite").read_bytes()
    assert FAKE_TOKEN.encode() not in ledger_bytes
