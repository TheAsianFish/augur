"""The leakage gate, and the scan that proves a ledger obeys it.

A forecast may only read a source whose date is *proven* to be at or before
the question's cutoff (live: now; backtest: the open date), and in a backtest
only a model released before the question opened may forecast it. The gate
decides at run time; :func:`scan_ledger` re-checks every stored row, so a
ledger written by buggy code (or edited by hand) still fails the build.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
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
    that question), ``model_too_new`` (a backtest call by a model not
    released before the question opened), ``bad_cutoff`` (a cutoff later than
    the moment the question was recorded, or in a backtest later than the
    question's open time: every other check trusts the cutoff, so it is
    checked too), ``bad_timestamp`` (a cutoff, open, recorded or source time
    not in ``clock.to_iso``'s fixed-width form: the triggers and the checks
    here compare times as text, which is only time order in that form), and
    ``missing_guard`` (the ledger's leakage triggers are gone, so rows were not
    checked when written), and ``below_floor`` (a forecast submitted to a scored
    tournament after a member was answered by a model outside the frontier tier
    or on a host not trusted to serve it: the submit floor, checked on what the
    ledger says actually answered), and ``late_price`` (a market price dated
    after the instant it was asked for, or a market source whose date is not
    its price's: the source's date is what the other checks compare).
    """
    uri = f"file:{path}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        return _scan(conn)


GUARD_TRIGGERS = (
    "source_uses_no_late_source",
    "source_uses_no_undated_in_backtest",
    "call_sources_admitted_only",
)


# The shape ``clock.to_iso`` writes. Text comparison is time order only here.
ISO_GLOB = (
    "[0-9][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-6][0-9]"
    ".[0-9][0-9][0-9][0-9][0-9][0-9]Z"
)

_TIMESTAMP_COLUMNS = (
    # (table, column, nullable, how to name the row)
    ("questions", "cutoff", False, "run_id, question_id"),
    ("questions", "open_time", False, "run_id, question_id"),
    ("questions", "recorded_at", False, "run_id, question_id"),
    ("sources", "published_at", True, "'', 0"),
)


def _scan(conn: sqlite3.Connection) -> list[Violation]:
    found: list[Violation] = []
    for table, column, nullable, ident in _TIMESTAMP_COLUMNS:
        null_ok = f"{column} IS NOT NULL AND " if nullable else ""
        rows = conn.execute(
            f"SELECT {ident}, quote({column}) FROM {table} "  # noqa: S608 - fixed names
            f"WHERE {null_ok}({column} IS NULL OR typeof({column}) != 'text' "
            f"OR {column} NOT GLOB ?)",
            (ISO_GLOB,),
        )
        for run_id, qid, value in rows:
            found.append(
                Violation(
                    "bad_timestamp",
                    run_id,
                    qid,
                    f"{table}.{column} = {value} is not a fixed-width UTC timestamp, "
                    "so text order is not time order and the other checks cannot be trusted",
                )
            )
    present = {
        name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
    }
    for name in GUARD_TRIGGERS:
        if name not in present:
            found.append(Violation("missing_guard", "", 0, f"trigger {name} is missing"))
    cutoffs = conn.execute(
        """
        SELECT q.run_id, q.question_id, q.cutoff, q.recorded_at, q.open_time
        FROM questions q JOIN runs r ON r.run_id = q.run_id
        WHERE q.cutoff > q.recorded_at OR (r.mode = 'backtest' AND q.cutoff > q.open_time)
        """
    )
    for run_id, qid, cutoff, recorded, opened in cutoffs:
        found.append(
            Violation(
                "bad_cutoff",
                run_id,
                qid,
                f"cutoff {cutoff} is after the question was recorded ({recorded})"
                if cutoff > recorded
                else f"backtest cutoff {cutoff} is after the question opened ({opened})",
            )
        )
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
    found.extend(_scan_prices(conn))
    found.extend(_scan_floor(conn))
    return found


def _scan_prices(conn: sqlite3.Connection) -> list[Violation]:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'market_quotes'"
    ).fetchone()
    if exists is None:
        return []
    found = []
    rows = conn.execute(
        """
        SELECT m.source_id, m.platform, m.price_time, m.as_of, s.published_at
        FROM market_quotes m LEFT JOIN sources s ON s.source_id = m.source_id
        WHERE m.price_time NOT GLOB ? OR m.as_of NOT GLOB ?
           OR m.price_time > m.as_of
           OR s.published_at IS NULL OR s.published_at != m.price_time
        """,
        (ISO_GLOB, ISO_GLOB),
    )
    for sid, platform, price_time, as_of, published in rows:
        found.append(
            Violation(
                "late_price",
                "",
                0,
                f"{platform} price {sid} dated {price_time}, asked for as of {as_of}, "
                f"source dated {published}",
            )
        )
    return found


def _scan_floor(conn: sqlite3.Connection) -> list[Violation]:
    from augur.llm.registry import (
        FRONTIER_HOSTS,
        FRONTIER_TIER,
        PRACTICE_TOURNAMENTS,
        bare_model_name,
    )

    columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_calls)")}
    host = "c.endpoint_host" if "endpoint_host" in columns else "NULL"
    marks = ", ".join("?" for _ in PRACTICE_TOURNAMENTS)
    rows = conn.execute(
        f"""
        SELECT DISTINCT s.run_id, s.question_id, c.model, {host}
        FROM submissions s
        JOIN runs r ON r.run_id = s.run_id
        JOIN forecasts f ON f.run_id = s.run_id AND f.question_id = s.question_id
            AND f.kind = 'member'
        JOIN llm_calls c ON c.call_id = f.call_id
        WHERE s.kind = 'forecast' AND s.status = 'submitted'
          AND r.tournament NOT IN ({marks})
        """,  # noqa: S608 - fixed column name and placeholders
        tuple(sorted(PRACTICE_TOURNAMENTS)),
    )
    found = []
    for run_id, qid, model, endpoint_host in rows:
        if bare_model_name(model) in FRONTIER_TIER and endpoint_host in FRONTIER_HOSTS:
            continue
        found.append(
            Violation(
                "below_floor",
                run_id,
                qid,
                f"submitted to a scored tournament from {model} "
                f"on {endpoint_host or 'an unknown host'}",
            )
        )
    return found
