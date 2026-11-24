"""The shadow round: what the shadow ledger already holds (M11).

A shadow tick forecasts unpublished, so Metaculus never learns that a
question was forecast and ``my_forecasts`` stays empty. What a tick must not
repeat is therefore read from the shadow ledger itself: the ledger rebuilt
from every shadow export on the ``ledger`` branch (``augur ledger import``).

- :func:`held_questions`: questions with an Augur aggregate forecast; the
  runner skips these (``augur forecast --shadow-ledger``) before any model call.
- :func:`questions_missing_template`: questions Augur forecast that have no
  template row yet and are still open, so a template run cut off by a timeout
  is caught up on the next tick (``augur baseline --shadow-ledger``).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from augur.log import get_logger
from augur.questions import Question, question_from_post

log = get_logger("augur.shadow")


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def held_questions(path: Path) -> frozenset[int]:
    """Question ids with an Augur aggregate forecast in the ledger at ``path``.

    A missing file holds nothing (the first tick of a season)."""
    if not path.exists():
        return frozenset()
    with closing(_connect(path)) as conn:
        if not _has_table(conn, "forecasts"):
            return frozenset()
        rows = conn.execute(
            "SELECT DISTINCT question_id FROM forecasts WHERE kind = 'aggregate'"
        ).fetchall()
    return frozenset(int(r[0]) for r in rows)


def templated_questions(path: Path) -> frozenset[int]:
    """Question ids with a template reference forecast in the ledger at ``path``."""
    if not path.exists():
        return frozenset()
    with closing(_connect(path)) as conn:
        if not _has_table(conn, "reference_forecasts"):
            return frozenset()
        rows = conn.execute(
            "SELECT DISTINCT question_id FROM reference_forecasts WHERE forecaster = 'template'"
        ).fetchall()
    return frozenset(int(r[0]) for r in rows)


def questions_missing_template(
    path: Path, now: datetime, *, also_templated: frozenset[int] = frozenset()
) -> list[Question]:
    """Questions Augur forecast in ``path`` with no template row there (nor in
    ``also_templated``) that are still open at ``now``, soonest-closing first,
    each parsed from the latest payload the ledger holds for it."""
    if not path.exists():
        return []
    done = templated_questions(path) | also_templated
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT q.question_id, q.payload_json FROM questions q "
            "WHERE EXISTS (SELECT 1 FROM forecasts f WHERE f.question_id = q.question_id "
            "AND f.kind = 'aggregate') ORDER BY q.recorded_at, q.rowid"
        ).fetchall()
    latest: dict[int, str] = {}
    for qid, payload in rows:
        latest[int(qid)] = str(payload)
    found: list[Question] = []
    for qid, payload in latest.items():
        if qid in done:
            continue
        try:
            q = question_from_post(json.loads(payload))
        except (ValueError, KeyError, TypeError) as exc:
            log.warning("shadow_unparsable_payload", question_id=qid, error=str(exc))
            continue
        if q.scheduled_close_time > now:
            found.append(q)
    found.sort(key=lambda q: (q.scheduled_close_time, q.question_id))
    return found
