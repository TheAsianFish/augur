"""Resolved questions of past Metaculus tournaments, for backtests on real questions.

``load_tournament`` lists a tournament's resolved posts and keeps the
questions a backtest may score. Eligibility, applied in this order, with the
count each step removes (mirroring :mod:`augur.forecastbench`):

1. *a single question post*: groups and conditionals carry no one question;
2. *a wanted type* (binary by default);
3. *not annulled or ambiguous*: those have no outcome;
4. *resolved, and the resolution visible*: a bot token sees
   ``resolution: null`` on some resolved questions (see :mod:`augur.resolve`);
5. *an open time*;
6. *resolved after it opened*: an actual resolve time before the open time is
   a data error;
7. *opened after the model's release*: the question must open strictly after
   the release date, so the model cannot remember the outcome;
8. *a resolution that parses* (``yes`` or ``no`` for binary).

A loaded set is committed as JSON (:func:`snapshot`) and re-read offline by
:func:`load_snapshot`, which applies the release filter again at read time.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from augur.clock import ensure_utc, parse_iso, to_iso
from augur.errors import AugurError
from augur.metaculus import MetaculusClient
from augur.questions import Question, question_from_post

SNAPSHOT_VERSION = 1
SOURCE = "Metaculus API, public question fields"
_ANNULLED = frozenset({"annulled", "ambiguous"})
_BINARY = {"yes": 1, "no": 0}

R_GROUP = "not a single question post (group or conditional)"
R_TYPE = "type not wanted"
R_ANNULLED = "annulled or ambiguous"
R_NOT_RESOLVED = "not resolved (or no actual resolve time)"
R_HIDDEN = "resolved, but the API shows this account no resolution"
R_NO_OPEN = "no open time"
R_NO_CLOSE = "no close time"
R_BEFORE_OPEN = "resolved before it opened"
R_RELEASE = "opened on or before the model's release"
R_UNPARSED = "resolution not parseable"


class ResolvedQuestion(BaseModel):
    """A question as it was posed, with what it resolved to."""

    model_config = ConfigDict(frozen=True)

    question: Question
    outcome: int
    resolution: str
    resolved_at: datetime
    tournament: str
    tournament_id: int | None = None


@dataclass
class ResolvedLoadReport:
    tournament: str
    released: date
    total: int = 0
    removed: dict[str, int] = field(default_factory=dict)
    kept: list[ResolvedQuestion] = field(default_factory=list)

    def remove(self, reason: str) -> None:
        self.removed[reason] = self.removed.get(reason, 0) + 1

    def lines(self) -> list[str]:
        out = [f"{self.tournament}: {self.total} resolved posts, model released {self.released}"]
        out.extend(f"  removed {n:>4}: {reason}" for reason, n in self.removed.items())
        out.append(f"  kept    {len(self.kept):>4}")
        return out


def _instant(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return ensure_utc(parse_iso(value))
    except ValueError:
        return None


def parse_outcome(qtype: str, resolution: str) -> int | None:
    """0 or 1 for a binary resolution, ``None`` when it is not one."""
    if qtype != "binary":
        return None
    return _BINARY.get(resolution.strip().lower())


def _judge(
    post: dict[str, Any],
    *,
    site: str,
    released: date,
    question_types: Collection[str],
    tournament: str,
    tournament_id: int | None,
) -> ResolvedQuestion | str:
    """The kept question, or the reason it is removed."""
    q = post.get("question")
    if not isinstance(q, dict):
        return R_GROUP
    if q.get("type") not in question_types:
        return R_TYPE
    raw = q.get("resolution")
    resolution = "" if raw is None else str(raw).strip()
    if resolution.lower() in _ANNULLED:
        return R_ANNULLED
    status = q.get("status") or post.get("status")
    resolved_at = _instant(q.get("actual_resolve_time"))
    if status != "resolved" or resolved_at is None:
        return R_NOT_RESOLVED
    if not resolution:
        return R_HIDDEN
    opened = _instant(q.get("open_time")) or _instant(post.get("open_time"))
    if opened is None:
        return R_NO_OPEN
    if resolved_at < opened:
        return R_BEFORE_OPEN
    if opened.date() <= released:
        return R_RELEASE
    outcome = parse_outcome(str(q.get("type")), resolution)
    if outcome is None:
        return R_UNPARSED
    try:
        question = question_from_post(post, site=site)
    except (ValueError, KeyError, TypeError):
        return R_NO_CLOSE
    return ResolvedQuestion(
        question=question.model_copy(update={"my_latest_forecast": None}),
        outcome=outcome,
        resolution=resolution,
        resolved_at=resolved_at,
        tournament=tournament,
        tournament_id=tournament_id,
    )


async def load_tournament(
    client: MetaculusClient,
    tournament: str | int,
    *,
    released: date,
    question_types: Collection[str] = ("binary",),
    limit: int | None = None,
) -> ResolvedLoadReport:
    """Resolved questions of ``tournament`` (slug or id) a model released on
    ``released`` cannot remember. Posts keep the API's order, so ``limit``
    always takes the same ones."""
    slug = str(tournament)
    posts = await client.list_posts(slug, statuses="resolved")
    report = ResolvedLoadReport(slug, released)
    tournament_id = int(tournament) if isinstance(tournament, int) or slug.isdigit() else None
    site = client.base_url.removesuffix("/api")
    for post in posts:
        report.total += 1
        verdict = _judge(
            post,
            site=site,
            released=released,
            question_types=question_types,
            tournament=slug,
            tournament_id=tournament_id,
        )
        if isinstance(verdict, str):
            report.remove(verdict)
        else:
            report.kept.append(verdict)
    _apply_limit(report, limit)
    return report


def _apply_limit(report: ResolvedLoadReport, limit: int | None) -> None:
    if limit is not None and len(report.kept) > limit:
        report.removed["over the limit"] = len(report.kept) - limit
        report.kept = report.kept[:limit]


def snapshot(report: ResolvedLoadReport) -> dict[str, Any]:
    """The kept questions as JSON-ready data: public question fields only."""
    return {
        "version": SNAPSHOT_VERSION,
        "source": SOURCE,
        "tournament": report.tournament,
        "loaded_with_release": report.released.isoformat(),
        "total_resolved_posts": report.total,
        "removed": dict(report.removed),
        "questions": [
            {
                "question": json.loads(r.question.model_dump_json(exclude={"my_latest_forecast"})),
                "outcome": r.outcome,
                "resolution": r.resolution,
                "resolved_at": to_iso(r.resolved_at),
                "tournament": r.tournament,
                "tournament_id": r.tournament_id,
            }
            for r in report.kept
        ],
    }


def write_snapshot(report: ResolvedLoadReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot(report), indent=1, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load_snapshot(path: Path, released: date, *, limit: int | None = None) -> ResolvedLoadReport:
    """Re-read a committed snapshot with the release filter for ``released``."""
    try:
        data: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        entries = data["questions"]
        report = ResolvedLoadReport(str(data["tournament"]), released)
        if data.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"version {data.get('version')!r}, expected {SNAPSHOT_VERSION}")
        for entry in entries:
            report.total += 1
            item = ResolvedQuestion.model_validate(entry)
            if item.question.open_time.date() <= released:
                report.remove(R_RELEASE)
            else:
                report.kept.append(item)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AugurError(
            f"cannot read the resolved-question snapshot {path} ({exc}); "
            "regenerate it with load_tournament and write_snapshot"
        ) from None
    _apply_limit(report, limit)
    return report
