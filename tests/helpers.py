"""Builders shared by tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

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
