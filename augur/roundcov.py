"""How much of a round the shadow schedule reached, by the day each question opened (M17).

``augur floor --round R --ledger L --by-day``: for each UTC day a question of the round
opened, the questions that opened and how many of them the ledgers hold an Augur
aggregate, a template, a ``stat`` and a ``blend`` row for, and all four. It counts; it
decides nothing (the floor rule is ``augur floor`` without ``--by-day``).

A question's opening day is the snapshot's ``opens_at`` (snapshots written from M17 on),
else the ``open_time`` a ledger recorded for it, else ``unknown``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from augur.standings import RoundStandings

UNKNOWN = "unknown"
ROLES = ("augur", "template", "stat", "blend")


@dataclass
class DayRow:
    day: str
    opened: int = 0
    held: dict[str, int] = field(default_factory=lambda: dict.fromkeys(ROLES, 0))
    all_four: int = 0

    def line(self) -> str:
        cells = ", ".join(f"{role} {self.held[role]}" for role in ROLES)
        return f"{self.day}: opened {self.opened}; {cells}; all four {self.all_four}"


def utc_day(stamp: str | None) -> str:
    """The UTC calendar day of an ISO 8601 instant (naive read as UTC); empty if unreadable."""
    if not stamp:
        return ""
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).date().isoformat()


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _held(paths: Sequence[Path]) -> tuple[dict[str, set[int]], dict[int, str]]:
    """Post ids holding each role's row, and each post's recorded opening day.

    Keyed by post, as the floor rule reads a round (its snapshot lists posts)."""
    held: dict[str, set[int]] = {role: set() for role in ROLES}
    opened: dict[int, str] = {}
    post_of: dict[int, int] = {}
    augur: set[int] = set()
    for path in paths:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            if _has_table(conn, "questions"):
                for qid, post_id, open_time in conn.execute(
                    "SELECT question_id, post_id, open_time FROM questions"
                ):
                    post_of.setdefault(int(qid), int(post_id))
                    day = utc_day(str(open_time) if open_time else None)
                    if day:
                        opened.setdefault(int(post_id), day)
            if _has_table(conn, "forecasts"):
                for (qid,) in conn.execute(
                    "SELECT DISTINCT f.question_id FROM forecasts f "
                    "JOIN runs r ON r.run_id = f.run_id "
                    "WHERE f.kind = 'aggregate' AND r.mode != 'unrestricted'"
                ):
                    augur.add(int(qid))
            if _has_table(conn, "reference_forecasts"):
                for post_id, who in conn.execute(
                    "SELECT DISTINCT post_id, forecaster FROM reference_forecasts "
                    "WHERE forecaster IN ('template', 'stat', 'blend')"
                ):
                    held[str(who)].add(int(post_id))
    held["augur"] = {post_of[q] for q in augur if q in post_of}
    return held, opened


def by_day(snapshot: RoundStandings, ledger_paths: Sequence[Path]) -> list[DayRow]:
    """One row per opening day, in day order (``unknown`` last), then a ``total`` row."""
    held, recorded = _held(ledger_paths)
    rows: dict[str, DayRow] = {}
    total = DayRow("total")
    for q in snapshot.questions:
        day = utc_day(q.opens_at) or recorded.get(q.post_id, "") or UNKNOWN
        row = rows.setdefault(day, DayRow(day))
        has = {role: q.post_id in held[role] for role in ROLES}
        for r in (row, total):
            r.opened += 1
            for role in ROLES:
                r.held[role] += has[role]
            r.all_four += all(has.values())
    ordered = sorted(rows.values(), key=lambda r: (r.day == UNKNOWN, r.day))
    return [*ordered, total]


def lines(snapshot: RoundStandings, ledger_paths: Sequence[Path]) -> list[str]:
    head = (
        f"{snapshot.round.slug} (snapshot {snapshot.fetched_at:%Y-%m-%dT%H:%MZ}): questions "
        "by the UTC day they opened, and how many the ledgers hold each row for "
        "(a count, not a verdict)"
    )
    return [head, *(f"  {row.line()}" for row in by_day(snapshot, ledger_paths))]
