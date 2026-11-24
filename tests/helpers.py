"""Builders shared by tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from augur.ledger import Ledger
from augur.llm.catalog import Catalog
from augur.llm.endpoints import Route
from augur.questions import Question, question_from_post
from augur.research.quotes import MarketQuote, Match
from augur.research.source import DateEvidence, Source
from augur.standings import RoundInfo, RoundQuestion, RoundStandings


def binary_post(
    post_id: int = 100,
    question_id: int = 101,
    *,
    opened: datetime,
    hours_open: float = 3,
    title: str = "Will the test pass?",
) -> dict[str, Any]:
    close = opened + timedelta(hours=hours_open)
    return {
        "id": post_id,
        "title": title,
        "question": {
            "id": question_id,
            "title": title,
            "type": "binary",
            "status": "open",
            "open_time": opened.isoformat().replace("+00:00", "Z"),
            "scheduled_close_time": close.isoformat().replace("+00:00", "Z"),
            "description": "Background.",
            "resolution_criteria": "Resolves Yes if it passes.",
            "fine_print": "",
        },
    }


def binary_question(opened: datetime, **kwargs: Any) -> tuple[Question, dict[str, Any]]:
    post = binary_post(opened=opened, **kwargs)
    return question_from_post(post), post


def source(
    published_at: datetime | None,
    *,
    retrieved_at: datetime,
    evidence: DateEvidence = DateEvidence.REVISION_TIMESTAMP,
    text: str = "Some text about the question.",
    url: str = "https://en.wikipedia.org/w/index.php?oldid=1",
) -> Source:
    return Source.build(
        searcher="wikipedia",
        query="test",
        url=url,
        title="Test",
        excerpt=text,
        published_at=published_at,
        retrieved_at=retrieved_at,
        as_of=retrieved_at,
        date_evidence=evidence,
        evidence_detail="test fixture",
    )


def market_source(
    price_time: datetime,
    *,
    as_of: datetime,
    probability: float = 0.3,
    match: Match = "exact",
) -> Source:
    """A Manifold price as of ``as_of``, set by a bet at ``price_time``."""
    quote = MarketQuote(
        platform="manifold",
        market_id="abc123",
        title="Will the test market resolve Yes?",
        url="https://manifold.markets/test/will-the-test-market-resolve-yes",
        outcome="Yes",
        probability=probability,
        price_time=price_time,
        as_of=as_of,
        match=match,
    )
    return Source.build(
        searcher="markets",
        query="test market",
        url=quote.url,
        title=quote.title,
        excerpt=f"Manifold price {probability:.0%}",
        published_at=price_time,
        retrieved_at=as_of,
        as_of=as_of,
        date_evidence=DateEvidence.MARKET_TRADE,
        evidence_detail="last bet at or before as_of (test fixture)",
        quote=quote,
    )


def catalog_of(*models: tuple[str, float, float], listed: datetime | None = None) -> Catalog:
    """A model list with ``(id, usd per million in, usd per million out)`` entries."""
    created = int((listed or datetime(2025, 1, 1, tzinfo=UTC)).timestamp())
    return Catalog.from_api(
        {
            "data": [
                {
                    "id": model,
                    "created": created,
                    "pricing": {"prompt": f"{inp / 1e6:.12f}", "completion": f"{out / 1e6:.12f}"},
                }
                for model, inp, out in models
            ]
        },
        datetime(2026, 11, 18, tzinfo=UTC),
    )


def free_catalog(*routes: Route) -> Catalog:
    """Every ``openrouter`` route listed at $0, so the cost caps never bind in a test."""
    return catalog_of(*((r.model, 0.0, 0.0) for r in routes))


def record_aggregate(
    ledger: Ledger, run_id: str, question: Question, values: dict[str, Any]
) -> None:
    """Augur's aggregate forecast for ``question`` in ``run_id``."""
    ledger.record_forecast(
        run_id=run_id,
        question_id=question.question_id,
        kind="aggregate",
        member=None,
        method="median",
        probability=values.get("probability"),
        reasoning="r",
        call_id=None,
        question_type=question.type,
        values=values,
    )


def record_template(ledger: Ledger, question: Question, values: dict[str, Any]) -> None:
    """The shadow template's forecast for ``question``, as ``augur baseline`` records it."""
    ledger.record_reference(
        question_id=question.question_id,
        post_id=question.post_id,
        question_type=question.type,
        forecaster="template",
        detail="shadow template (test fixture)",
        values=values,
        as_of=datetime(2026, 11, 30, 12, tzinfo=UTC),
        source="test",
    )


def leaderboard_entry(username: str, average: float, coverage: float) -> dict[str, Any]:
    """A leaderboard row whose spot peer score per question covered is ``average``."""
    score = average * coverage
    return {
        "user": {"id": 1, "username": username, "is_bot": True},
        "aggregation_method": None,
        "score": score,
        "ci_lower": score - 10 * coverage,
        "ci_upper": score + 10 * coverage,
        "rank": 1,
        "excluded": False,
        "exclusion_status": 0,
        "coverage": coverage,
        "contribution_count": coverage,
    }


def round_snapshot(
    post_ids: list[int],
    *,
    entries: list[dict[str, Any]] | None = None,
    max_coverage: float | None = None,
    fetched_at: datetime = datetime(2026, 12, 13, 12, tzinfo=UTC),
    slug: str = "minibench-2026-11-30",
    finalized: bool = False,
) -> RoundStandings:
    """A MiniBench round's standings: its posts and a primary leaderboard."""
    return RoundStandings(
        round=RoundInfo(id=33200, slug=slug, name="MiniBench test round"),
        leaderboard_raw={
            "project_slug": slug,
            "score_type": "spot_peer_score",
            "finalized": finalized,
            "max_coverage": float(len(post_ids)) if max_coverage is None else max_coverage,
            "is_primary_leaderboard": True,
            "entries": entries or [],
        },
        questions=[
            RoundQuestion(
                post_id=p, question_id=None, title=f"Post {p}", status="resolved", type=None
            )
            for p in post_ids
        ],
        fetched_at=fetched_at,
        source_url="https://example.test/api/leaderboards/project/33200/",
    )
