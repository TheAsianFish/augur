"""A searcher that returns fixed sources, one of them dated after any sane cutoff."""

from __future__ import annotations

from datetime import datetime, timedelta

from augur.research.source import DateEvidence, Source


class FixedSearcher:
    name = "fixed"

    def __init__(self, *, late_by: timedelta = timedelta(days=1), fail: bool = False) -> None:
        self.late_by = late_by
        self.fail = fail
        self.calls: list[tuple[str, datetime]] = []

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        from augur.errors import SearchError

        self.calls.append((query, as_of))
        if self.fail:
            raise SearchError(f"fixed searcher down for {query!r}")
        good = Source.build(
            searcher=self.name,
            query=query,
            url=f"https://example.org/{query.replace(' ', '-')}",
            title=f"Background on {query}",
            excerpt=f"Facts about {query}.",
            published_at=as_of - timedelta(days=3),
            retrieved_at=as_of,
            as_of=as_of,
            date_evidence=DateEvidence.REVISION_TIMESTAMP,
            evidence_detail="fixed",
        )
        late = Source.build(
            searcher=self.name,
            query=query,
            url="https://example.org/the-answer",
            title="The answer, written after the question closed",
            excerpt="It resolved Yes.",
            published_at=as_of + self.late_by,
            retrieved_at=as_of,
            as_of=as_of,
            date_evidence=DateEvidence.PROVIDER_PUBLISHED_BOUNDED,
            evidence_detail="fixed (buggy provider)",
        )
        return [good, late][:limit]
