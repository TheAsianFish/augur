"""Builders shared by tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from augur.llm.catalog import Catalog
from augur.llm.endpoints import Route
from augur.questions import Question, question_from_post
from augur.research.source import DateEvidence, Source


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
