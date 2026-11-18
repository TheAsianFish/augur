"""The private comment posted with each forecast.

Built only from what produced the forecast: the members' own reasoning, the
research brief they read, and the sources it cites with their proven dates.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from augur.agents.forecaster import MemberForecast
from augur.agents.researcher import ResearchResult
from augur.predictions import BinaryPrediction, Prediction, summary
from augur.questions import Question
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


def _source_line(label: str, s: Source) -> str:
    when = (
        f"{_EVIDENCE[s.date_evidence]} {s.published_at:%Y-%m-%d}" if s.published_at else "undated"
    )
    return f"- [{label}] [{s.title}]({s.url}) ({s.searcher}, {when})"


_HOW = {
    "binary": "the median of",
    "multiple_choice": "the per-option median, normalised, of",
    "numeric": "the average of the CDFs of",
    "discrete": "the average of the CDFs of",
}


def render_comment(
    *,
    q: Question,
    prediction: Prediction,
    members: Sequence[MemberForecast],
    research: ResearchResult,
    cutoff: datetime,
    requested: int,
) -> str:
    """Markdown for the private comment. Pure: the same inputs give the same text."""
    ordered = sorted(members, key=lambda m: m.member)
    models = sorted({m.route.model for m in ordered})
    model = " and ".join(models) if models else "no model"
    how = _HOW.get(q.type, "combined from")
    if isinstance(prediction, BinaryPrediction):
        spread = " (" + ", ".join(summary(q, m.prediction) for m in ordered) + ")"
    else:
        spread = ""
    lines = [
        f"**Forecast: {summary(q, prediction)}**, {how} {len(ordered)} independent "
        f"forecasts{spread} by {model}"
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
            f"{m.member + 1}. ({summary(q, m.prediction)}) "
            f"{_clip(m.reasoning, MAX_REASONING_CHARS)}"
        )
    if research.sources:
        lines += ["", "**Sources**", ""]
        lines += [_source_line(f"S{i}", s) for i, s in enumerate(research.sources, start=1)]
    text = "\n".join(lines).strip()
    if len(text) > MAX_COMMENT_CHARS:
        text = text[: MAX_COMMENT_CHARS - 2].rsplit("\n", 1)[0] + "\n…"
    return text
