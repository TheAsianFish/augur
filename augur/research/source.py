"""The ``Source`` record: what a model may read, and the evidence for its date."""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, field_validator

from augur.clock import ensure_utc, to_iso


class DateEvidence(StrEnum):
    """How a source's date is proven. Only proof, never a page's own metadata, admits it."""

    REVISION_TIMESTAMP = "revision_timestamp"
    """A wiki revision's server-side save time."""
    ARCHIVE_CAPTURE = "archive_capture"
    """An archive's capture timestamp."""
    PROVIDER_PUBLISHED_BOUNDED = "provider_published_bounded"
    """A news provider's publish date, from a query bounded by ``end_timestamp <= as_of``."""
    NONE = "none"
    """No proof (for example a date read from the page itself)."""


class Source(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str
    searcher: str
    query: str
    url: str
    title: str
    excerpt: str
    content_sha256: str
    published_at: datetime | None
    retrieved_at: datetime
    as_of: datetime
    date_evidence: DateEvidence
    evidence_detail: str

    @field_validator("published_at", "retrieved_at", "as_of")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else ensure_utc(value)

    @classmethod
    def build(
        cls,
        *,
        searcher: str,
        query: str,
        url: str,
        title: str,
        excerpt: str,
        published_at: datetime | None,
        retrieved_at: datetime,
        as_of: datetime,
        date_evidence: DateEvidence,
        evidence_detail: str,
    ) -> Source:
        """Construct a source whose id is a hash of what it says and when."""
        content_sha = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        stamp = to_iso(published_at) if published_at else "undated"
        key = "\n".join([searcher, url, stamp, content_sha])
        source_id = "src_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return cls(
            source_id=source_id,
            searcher=searcher,
            query=query,
            url=url,
            title=title,
            excerpt=excerpt,
            content_sha256=content_sha,
            published_at=published_at,
            retrieved_at=retrieved_at,
            as_of=as_of,
            date_evidence=date_evidence,
            evidence_detail=evidence_detail,
        )


class Searcher(Protocol):
    """Anything that can look things up as they stood at an instant."""

    name: str

    async def search(self, query: str, as_of: datetime, limit: int) -> list[Source]:
        """Sources relevant to ``query`` that can be dated at or before ``as_of``."""
        ...
