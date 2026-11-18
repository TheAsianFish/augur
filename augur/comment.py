"""The private comment posted with each forecast.

Built only from what produced the forecast: the members' own reasoning, the
research brief they read, and the sources it cites with their proven dates.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from augur.agents.forecaster import MemberForecast
from augur.agents.researcher import ResearchResult
from augur.research.source import DateEvidence, Source

MAX_COMMENT_CHARS = 9000
MAX_REASONING_CHARS = 700
MAX_BRIEF_CHARS = 1800

_EVIDENCE = {
    DateEvidence.REVISION_TIMESTAMP: "revision of",
    DateEvidence.ARCHIVE_CAPTURE: "captured",
    DateEvidence.PROVIDER_PUBLISHED_BOUNDED: "published",
    DateEvidence.NONE: "dated (unproven)",
}


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def _pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def _source_line(label: str, s: Source) -> str:
    when = (
        f"{_EVIDENCE[s.date_evidence]} {s.published_at:%Y-%m-%d}" if s.published_at else "undated"
    )
    return f"- [{label}] [{s.title}]({s.url}) ({s.searcher}, {when})"


def render_comment(
    *,
    probability: float,
    members: Sequence[MemberForecast],
    research: ResearchResult,
    model: str,
    cutoff: datetime,
    requested: int,
) -> str:
    """Markdown for the private comment. Pure: the same inputs give the same text."""
    ordered = sorted(members, key=lambda m: m.member)
    spread = ", ".join(_pct(m.probability) for m in ordered)
    lines = [
        f"**Forecast: {_pct(probability)}**, the median of {len(ordered)} independent "
        f"forecasts ({spread}) by {model}"
        + (
            f"; {requested - len(ordered)} of {requested} draws failed."
            if len(ordered) < requested
            else "."
        ),
        "",
        f"**Research** (only sources dated at or before {cutoff:%Y-%m-%d %H:%M} UTC):",
        "",
        _clip(research.brief, MAX_BRIEF_CHARS),
        "",
        "**Each forecaster's reasoning**",
        "",
    ]
    for m in ordered:
        lines.append(
            f"{m.member + 1}. ({_pct(m.probability)}) {_clip(m.reasoning, MAX_REASONING_CHARS)}"
        )
    if research.sources:
        lines += ["", "**Sources**", ""]
        lines += [_source_line(f"S{i}", s) for i, s in enumerate(research.sources, start=1)]
    text = "\n".join(lines).strip()
    if len(text) > MAX_COMMENT_CHARS:
        text = text[: MAX_COMMENT_CHARS - 2].rsplit("\n", 1)[0] + "\n…"
    return text
