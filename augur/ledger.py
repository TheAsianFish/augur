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
    "model_probes",
    "limit_tests",
    "model_switches",
    "question_outcomes",
    "resolutions",
    "reference_forecasts",
    "metaculus_scores",
    "market_quotes",
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
        pending = [(v, sql) for v, sql in _migrations() if v > current]
        if not pending:
            self._install_append_only_guards()
            return
        # A migration may rebuild a table (SQLite cannot change a CHECK in
        # place): foreign keys are off while it runs and checked after, and
        # a rename keeps the triggers that name the table pointing at it.
        # Both pragmas are no-ops inside a transaction, so they are set here.
        self._conn.execute("PRAGMA foreign_keys = OFF")
        self._conn.execute("PRAGMA legacy_alter_table = ON")
        try:
            self._apply(pending)
        finally:
            self._conn.execute("PRAGMA legacy_alter_table = OFF")
            self._conn.execute("PRAGMA foreign_keys = ON")
        self._install_append_only_guards()

    def _apply(self, pending: list[tuple[int, str]]) -> None:
        for version, sql in pending:
            try:
                self._conn.execute("BEGIN")
                for statement in _split_sql(sql):
                    self._conn.execute(statement)
                broken = self._conn.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise sqlite3.IntegrityError(f"{len(broken)} rows lost their foreign key")
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
        if source.quote is not None:
            self._record_quote(source)

    def _record_quote(self, source: Source) -> None:
        quote = source.quote
        assert quote is not None
        if source.published_at != quote.price_time:
            raise LedgerError(
                f"market source {source.source_id} is dated {source.published_at} but its price "
                f"at {quote.price_time}; a market source's date must be its price's"
            )
        row = {
            "source_id": source.source_id,
            "platform": quote.platform,
            "market_id": quote.market_id,
            "title": quote.title,
            "url": quote.url,
            "outcome": quote.outcome,
            "probability": quote.probability,
            "price_time": to_iso(quote.price_time),
            "as_of": to_iso(quote.as_of),
            "match": quote.match,
            "volume": quote.volume,
            "closes_at": to_iso(quote.closes_at) if quote.closes_at else None,
            "recorded_at": self._now(),
        }
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        self._conn.execute(
            f"INSERT OR IGNORE INTO market_quotes ({cols}) VALUES ({marks})",  # noqa: S608
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
        release_basis: str | None = None,
        endpoint_host: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_effort_applied: bool | None = None,
        served_model: str | None = None,
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
                    "endpoint_host": endpoint_host,
                    "served_model": served_model,
                    "release_basis": release_basis,
                    "reasoning_effort": reasoning_effort,
                    "reasoning_effort_applied": None
                    if reasoning_effort_applied is None
                    else int(reasoning_effort_applied),
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
        probability: float | None,
        reasoning: str,
        call_id: str | None,
        question_type: str = "binary",
        values: dict[str, Any] | None = None,
    ) -> str:
        """One member's or the aggregate's forecast. ``probability`` is binary only;
        ``values`` holds the full prediction (option probabilities, percentiles, CDF)."""
        forecast_id = new_id("fc")
        if values is None and probability is not None:
            values = {"probability": probability}
        self._insert(
            "forecasts",
            {
                "forecast_id": forecast_id,
                "run_id": run_id,
                "question_id": question_id,
                "kind": kind,
                "member": member,
                "method": method,
                "question_type": question_type,
                "probability": probability,
                "values_json": None if values is None else json.dumps(values, sort_keys=True),
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

    def record_probe(self, row: dict[str, Any]) -> str:
        """One ``model_probes`` row; ``row`` holds every column but the id and time."""
        probe_id = new_id("probe")
        self._insert("model_probes", {"probe_id": probe_id, **row, "probed_at": self._now()})
        return probe_id

    def record_limit_test(self, row: dict[str, Any]) -> str:
        test_id = new_id("limit")
        self._insert("limit_tests", {"test_id": test_id, **row})
        return test_id

    def record_outcome(
        self,
        *,
        run_id: str,
        question_id: int,
        post_id: int,
        question_type: str,
        status: str,
        reason: str,
        submitted: bool,
        closes_at: datetime | None,
        minutes_left: float | None,
        cost_usd: float,
        seconds: float | None,
    ) -> str:
        """What happened to one question a run saw: the run's coverage, row by row."""
        outcome_id = new_id("qo")
        self._insert(
            "question_outcomes",
            {
                "outcome_id": outcome_id,
                "run_id": run_id,
                "question_id": question_id,
                "post_id": post_id,
                "question_type": question_type,
                "status": status,
                "reason": reason,
                "submitted": int(submitted),
                "closes_at": to_iso(closes_at) if closes_at else None,
                "minutes_left": minutes_left,
                "cost_usd": cost_usd,
                "seconds": seconds,
                "recorded_at": self._now(),
            },
        )
        return outcome_id

    def record_switch(
        self,
        *,
        run_id: str,
        question_id: int | None,
        call_role: str,
        member: int | None,
        from_route: str | None,
        to_route: str | None,
        reason: str,
        outcome: str,
        detail: str,
    ) -> str:
        switch_id = new_id("sw")
        self._insert(
            "model_switches",
            {
                "switch_id": switch_id,
                "run_id": run_id,
                "question_id": question_id,
                "call_role": call_role,
                "member": member,
                "from_route": from_route,
                "to_route": to_route,
                "reason": reason,
                "outcome": outcome,
                "detail": detail[:500],
                "recorded_at": self._now(),
            },
        )
        return switch_id

    def record_resolution(
        self,
        *,
        question_id: int,
        post_id: int,
        question_type: str,
        status: str,
        value: str | None,
        resolved_at: datetime | None,
        source: str,
        evidence: str,
    ) -> str:
        """What a question resolved to (or that it was annulled). The latest row counts."""
        resolution_id = new_id("res")
        self._insert(
            "resolutions",
            {
                "resolution_id": resolution_id,
                "question_id": question_id,
                "post_id": post_id,
                "question_type": question_type,
                "status": status,
                "value": value,
                "resolved_at": to_iso(resolved_at) if resolved_at else None,
                "source": source,
                "evidence": evidence[:500],
                "recorded_at": self._now(),
            },
        )
        return resolution_id

    def record_reference(
        self,
        *,
        question_id: int,
        post_id: int,
        question_type: str,
        forecaster: str,
        detail: str,
        values: dict[str, Any],
        as_of: datetime,
        source: str,
    ) -> str:
        """Someone else's forecast on the same question: community, template or market."""
        reference_id = new_id("ref")
        probability = values.get("probability") if question_type == "binary" else None
        self._insert(
            "reference_forecasts",
            {
                "reference_id": reference_id,
                "question_id": question_id,
                "post_id": post_id,
                "question_type": question_type,
                "forecaster": forecaster,
                "detail": detail[:500],
                "probability": probability,
                "values_json": json.dumps(values, sort_keys=True),
                "as_of": to_iso(as_of),
                "source": source,
                "recorded_at": self._now(),
            },
        )
        return reference_id

    def record_metaculus_score(
        self, *, question_id: int, post_id: int, forecaster: str, score_type: str, score: float
    ) -> str:
        score_id = new_id("ms")
        self._insert(
            "metaculus_scores",
            {
                "score_id": score_id,
                "question_id": question_id,
                "post_id": post_id,
                "forecaster": forecaster,
                "score_type": score_type,
                "score": float(score),
                "recorded_at": self._now(),
            },
        )
        return score_id

    # -- reads ----------------------------------------------------------

    def latest_resolution(self, question_id: int) -> tuple[str, str | None] | None:
        """``(status, value)`` of the latest resolution row, or ``None``."""
        row = self._conn.execute(
            "SELECT status, value FROM resolutions WHERE question_id = ? "
            "ORDER BY recorded_at DESC, rowid DESC LIMIT 1",
            (question_id,),
        ).fetchone()
        return None if row is None else (str(row[0]), row[1])

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
