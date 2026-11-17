"""The leakage gate, and the scan that proves a ledger obeys it.

A forecast may only read a source whose date is *proven* to be at or before
the question's cutoff (live: now; backtest: the open date), and in a backtest
only a model released before the question opened may forecast it. The gate
decides at run time; :func:`scan_ledger` re-checks every stored row, so a
ledger written by buggy code (or edited by hand) still fails the build.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path

from augur.clock import ensure_utc
from augur.research.source import DateEvidence, Source


class RunMode(StrEnum):
    LIVE = "live"
    DRY_RUN = "dry_run"
    BACKTEST = "backtest"


@dataclass(frozen=True)
class Admission:
    source_id: str
    admitted: bool
    reason: str


def admit_source(source: Source, cutoff: datetime, mode: RunMode) -> Admission:
    """Decide whether a model may read ``source`` for a question cut off at ``cutoff``."""
    cutoff = ensure_utc(cutoff)
    sid = source.source_id
    if source.published_at is None:
        if mode is RunMode.BACKTEST:
            return Admission(sid, False, "undated source in a backtest")
        return Admission(sid, True, "undated; admitted because the run is live")
    if source.published_at > cutoff:
        return Admission(sid, False, f"dated {source.published_at.isoformat()} after the cutoff")
    if source.date_evidence is DateEvidence.NONE:
        if mode is RunMode.BACKTEST:
            return Admission(sid, False, "date not proven (no evidence) in a backtest")
        return Admission(sid, True, "date unproven; admitted because the run is live")
    return Admission(sid, True, f"dated at or before the cutoff by {source.date_evidence.value}")


def admit_model(release: date | None, question_open: datetime, mode: RunMode) -> Admission:
    """In a backtest a model must have been released strictly before the question opened."""
    key = "model"
    if mode is not RunMode.BACKTEST:
        return Admission(key, True, "live run: any model may forecast")
    if release is None:
        return Admission(key, False, "model release date unknown in a backtest")
    if release >= ensure_utc(question_open).date():
        return Admission(key, False, f"model released {release} on or after the question opened")
    return Admission(key, True, f"model released {release}, before the question opened")


@dataclass(frozen=True)
class Violation:
    kind: str
    run_id: str
    question_id: int
    detail: str


def scan_ledger(path: Path) -> list[Violation]:
    """Every leakage violation stored in the ledger at ``path``.

    Kinds: ``late_source`` (an admitted source dated after its cutoff),
    ``undated_source`` (an admitted undated or unproven source in a backtest),
    ``unadmitted_use`` (a model call cited a source the gate did not admit for
    that question), and ``model_too_new`` (a backtest call by a model not
    released before the question opened).
    """
    uri = f"file:{path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return _scan(conn)


def _scan(conn: sqlite3.Connection) -> list[Violation]:
    found: list[Violation] = []
    late = conn.execute(
        """
        SELECT u.run_id, u.question_id, s.source_id, s.published_at, q.cutoff
        FROM source_uses u
        JOIN sources s ON s.source_id = u.source_id
        JOIN questions q ON q.run_id = u.run_id AND q.question_id = u.question_id
        WHERE u.admitted = 1 AND s.published_at IS NOT NULL AND s.published_at > q.cutoff
        """
    )
    for run_id, qid, sid, published, cutoff in late:
        found.append(
            Violation("late_source", run_id, qid, f"{sid} dated {published} > cutoff {cutoff}")
        )
    undated = conn.execute(
        """
        SELECT u.run_id, u.question_id, s.source_id, s.date_evidence
        FROM source_uses u
        JOIN sources s ON s.source_id = u.source_id
        JOIN runs r ON r.run_id = u.run_id
        WHERE u.admitted = 1 AND r.mode = 'backtest'
          AND (s.published_at IS NULL OR s.date_evidence = 'none')
        """
    )
    for run_id, qid, sid, evidence in undated:
        found.append(
            Violation("undated_source", run_id, qid, f"{sid} has no proven date ({evidence})")
        )
    unadmitted = conn.execute(
        """
        SELECT c.run_id, c.question_id, cs.source_id
        FROM call_sources cs
        JOIN llm_calls c ON c.call_id = cs.call_id
        WHERE NOT EXISTS (
            SELECT 1 FROM source_uses u
            WHERE u.run_id = c.run_id AND u.question_id = c.question_id
              AND u.source_id = cs.source_id AND u.admitted = 1
        )
        """
    )
    for run_id, qid, sid in unadmitted:
        found.append(Violation("unadmitted_use", run_id, qid, f"{sid} reached a model unadmitted"))
    too_new = conn.execute(
        """
        SELECT c.run_id, c.question_id, c.model, c.model_release_date, q.open_time
        FROM llm_calls c
        JOIN runs r ON r.run_id = c.run_id
        JOIN questions q ON q.run_id = c.run_id AND q.question_id = c.question_id
        WHERE r.mode = 'backtest'
          AND (c.model_release_date IS NULL OR c.model_release_date >= substr(q.open_time, 1, 10))
        """
    )
    for run_id, qid, model, release, opened in too_new:
        found.append(
            Violation("model_too_new", run_id, qid, f"{model} released {release}, opened {opened}")
        )
    return found
