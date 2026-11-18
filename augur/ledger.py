"""The append-only SQLite ledger.

Every question, model call, source, gate decision, forecast and submission is
a row; nothing is updated or deleted (triggers refuse both), and a correction
is a new row. Schema changes are numbered SQL files in ``augur/migrations``.

SQLite writes here are local and take microseconds, so the ledger is
synchronous; all network I/O elsewhere is async.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import date, datetime
from importlib import resources
from pathlib import Path
from typing import Any, Self

from augur.clock import Clock, to_iso
from augur.errors import LedgerError
from augur.leakage import Admission, RunMode
from augur.questions import Question
from augur.research.source import Source

APPEND_ONLY_TABLES = (
    "runs",
    "questions",
    "sources",
    "source_uses",
    "llm_calls",
    "call_sources",
    "forecasts",
    "submissions",
    "schema_migrations",
)


def _migrations() -> list[tuple[int, str]]:
    found = []
    for entry in resources.files("augur.migrations").iterdir():
        name = entry.name
        if name.endswith(".sql") and name[:4].isdigit():
            found.append((int(name[:4]), entry.read_text(encoding="utf-8")))
    return sorted(found)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


class Ledger:
    """One SQLite file. Open with :meth:`open`; close when done."""

    def __init__(self, conn: sqlite3.Connection, clock: Clock) -> None:
        self._conn = conn
        self._clock = clock

    @classmethod
    def open(cls, path: Path | str, clock: Clock) -> Self:
        conn = sqlite3.connect(str(path), isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        # Without this, INSERT OR REPLACE deletes the old row without firing the
        # DELETE guard, and a run's mode could be rewritten in place.
        conn.execute("PRAGMA recursive_triggers = ON")
        if str(path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        ledger = cls(conn, clock)
        ledger._migrate()
        return ledger

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT max(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def _migrate(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        current = self.schema_version()
        for version, sql in _migrations():
            if version <= current:
                continue
            try:
                self._conn.execute("BEGIN")
                for statement in _split_sql(sql):
                    self._conn.execute(statement)
                self._conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, self._now()),
                )
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                raise LedgerError(
                    f"migration {version:04d} failed ({exc}); the ledger was left unchanged"
                ) from exc
        self._install_append_only_guards()

    def _install_append_only_guards(self) -> None:
        for table in APPEND_ONLY_TABLES:
            for verb in ("UPDATE", "DELETE"):
                self._conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {table}_no_{verb.lower()} "
                    f"BEFORE {verb} ON {table} BEGIN "
                    f"SELECT RAISE(ABORT, 'ledger is append-only: {verb} on {table}'); END"
                )

    def _now(self) -> str:
        return to_iso(self._clock.now())

    def _insert(self, table: str, row: dict[str, Any]) -> None:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        try:
            # Table and column names come from this module, never from input.
            sql = f"INSERT INTO {table} ({cols}) VALUES ({marks})"  # noqa: S608
            self._conn.execute(sql, tuple(row.values()))
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"ledger refused a row in {table}: {exc}") from exc

    # -- writes ---------------------------------------------------------

    def start_run(
        self,
        *,
        mode: RunMode,
        tournament: str,
        provider: str,
        model: str,
        settings: dict[str, object],
        code_version: str,
    ) -> str:
        run_id = new_id("run")
        self._insert(
            "runs",
            {
                "run_id": run_id,
                "started_at": self._now(),
                "mode": mode.value,
                "tournament": tournament,
                "provider": provider,
                "model": model,
                "settings_json": json.dumps(settings, sort_keys=True, default=str),
                "code_version": code_version,
            },
        )
        return run_id

    def record_question(
        self, run_id: str, question: Question, cutoff: datetime, payload: dict[str, Any]
    ) -> None:
        self._insert(
            "questions",
            {
                "run_id": run_id,
                "question_id": question.question_id,
                "post_id": question.post_id,
                "type": question.type,
                "title": question.title,
                "open_time": to_iso(question.open_time),
                "close_time": to_iso(question.scheduled_close_time),
                "cutoff": to_iso(cutoff),
                "payload_json": json.dumps(payload, sort_keys=True),
                "recorded_at": self._now(),
            },
        )

    def record_source(self, source: Source) -> None:
        """Store a source before any model sees it. Re-recording the same source is a no-op."""
        row = {
            "source_id": source.source_id,
            "searcher": source.searcher,
            "query": source.query,
            "url": source.url,
            "title": source.title,
            "excerpt": source.excerpt,
            "content_sha256": source.content_sha256,
            "published_at": to_iso(source.published_at) if source.published_at else None,
            "retrieved_at": to_iso(source.retrieved_at),
            "as_of": to_iso(source.as_of),
            "date_evidence": source.date_evidence.value,
            "evidence_detail": source.evidence_detail,
            "recorded_at": self._now(),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self._conn.execute(
            f"INSERT OR IGNORE INTO sources ({cols}) VALUES ({marks})",  # noqa: S608
            tuple(row.values()),
        )

    def record_admission(self, run_id: str, question_id: int, admission: Admission) -> None:
        try:
            self._insert(
                "source_uses",
                {
                    "run_id": run_id,
                    "question_id": question_id,
                    "source_id": admission.source_id,
                    "admitted": int(admission.admitted),
                    "reason": admission.reason,
                    "recorded_at": self._now(),
                },
            )
        except LedgerError as exc:
            if "leakage" in str(exc):
                from augur.errors import LeakageError

                raise LeakageError(str(exc)) from exc
            raise

    def record_call(
        self,
        *,
        run_id: str,
        question_id: int,
        role: str,
        member: int | None,
        provider: str,
        model: str,
        model_release_date: date | None,
        request_hash: str,
        request: dict[str, Any],
        response_text: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: float,
        latency_ms: int,
        error: str | None,
        started_at: datetime,
        source_ids: list[str],
    ) -> str:
        call_id = new_id("call")
        self._conn.execute("BEGIN")
        try:
            self._insert(
                "llm_calls",
                {
                    "call_id": call_id,
                    "run_id": run_id,
                    "question_id": question_id,
                    "role": role,
                    "member": member,
                    "provider": provider,
                    "model": model,
                    "model_release_date": model_release_date.isoformat()
                    if model_release_date
                    else None,
                    "request_hash": request_hash,
                    "request_json": json.dumps(request, sort_keys=True),
                    "response_text": response_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                    "error": error,
                    "started_at": to_iso(started_at),
                },
            )
            for source_id in source_ids:
                self._insert("call_sources", {"call_id": call_id, "source_id": source_id})
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return call_id

    def record_forecast(
        self,
        *,
        run_id: str,
        question_id: int,
        kind: str,
        member: int | None,
        method: str,
        probability: float,
        reasoning: str,
        call_id: str | None,
    ) -> str:
        forecast_id = new_id("fc")
        self._insert(
            "forecasts",
            {
                "forecast_id": forecast_id,
                "run_id": run_id,
                "question_id": question_id,
                "kind": kind,
                "member": member,
                "method": method,
                "probability": probability,
                "reasoning": reasoning,
                "call_id": call_id,
                "recorded_at": self._now(),
            },
        )
        return forecast_id

    def record_submission(
        self,
        *,
        run_id: str,
        question_id: int,
        post_id: int,
        kind: str,
        payload: dict[str, Any] | list[Any],
        status: str,
        http_status: int | None,
        response_excerpt: str | None,
    ) -> str:
        submission_id = new_id("sub")
        self._insert(
            "submissions",
            {
                "submission_id": submission_id,
                "run_id": run_id,
                "question_id": question_id,
                "post_id": post_id,
                "kind": kind,
                "payload_json": json.dumps(payload, sort_keys=True),
                "status": status,
                "http_status": http_status,
                "response_excerpt": response_excerpt,
                "recorded_at": self._now(),
            },
        )
        return submission_id

    # -- reads ----------------------------------------------------------

    def unposted_comment(self, question_id: int) -> str | None:
        """The comment that belongs to this account's forecast and was never posted.

        Call it only when the server shows this account's forecast standing.
        The text comes from the latest live run that attempted the forecast
        (posted, or failed in a way that may still have been processed), and
        only from that run's failed comment: never a dry run's text, which may
        say another probability. ``None`` when a comment was already posted,
        or no such run left one.
        """
        posted = self._conn.execute(
            "SELECT count(*) FROM submissions WHERE question_id = ? AND kind = 'comment' "
            "AND status = 'submitted'",
            (question_id,),
        ).fetchone()[0]
        if posted:
            return None
        forecast = self._conn.execute(
            "SELECT run_id FROM submissions WHERE question_id = ? AND kind = 'forecast' "
            "AND status IN ('submitted', 'failed') ORDER BY recorded_at DESC LIMIT 1",
            (question_id,),
        ).fetchone()
        if forecast is None:
            return None
        row = self._conn.execute(
            "SELECT payload_json FROM submissions WHERE question_id = ? AND run_id = ? "
            "AND kind = 'comment' AND status = 'failed' ORDER BY recorded_at DESC LIMIT 1",
            (question_id, forecast[0]),
        ).fetchone()
        if row is None:
            return None
        text = json.loads(row[0]).get("text")
        return text if isinstance(text, str) and text.strip() else None

    def count(self, table: str, run_id: str | None = None) -> int:
        if table not in APPEND_ONLY_TABLES:
            raise LedgerError(f"unknown table {table!r}")
        if run_id is None:
            row = self._conn.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608
        else:
            row = self._conn.execute(
                f"SELECT count(*) FROM {table} WHERE run_id = ?",  # noqa: S608
                (run_id,),
            ).fetchone()
        return int(row[0])


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements, keeping trigger bodies whole."""
    statements: list[str] = []
    buffer: list[str] = []
    for line in script.splitlines():
        stripped = line.strip()
        if not buffer and (not stripped or stripped.startswith("--")):
            continue
        buffer.append(line)
        text = "\n".join(buffer)
        if sqlite3.complete_statement(text):
            statements.append(text)
            buffer = []
    if any(part.strip() for part in buffer):
        raise LedgerError("migration ends with an incomplete statement")
    return statements
