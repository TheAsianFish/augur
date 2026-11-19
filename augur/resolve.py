"""``augur resolve``: append what each forecast question resolved to.

Three sources, each a new ``resolutions`` row (never an edit):

- **Metaculus**, for the questions in a ledger's live and dry runs: the
  question's ``resolution`` (or annulment). With it, anything else the API
  exposes for that question is kept for scoring: the community prediction's
  latest aggregate (``reference_forecasts``, forecaster ``community``) and
  Metaculus's own scores of this account's forecast and of the aggregate
  (``metaculus_scores``), which ``augur score --check`` compares with
  Augur's. Measured 2026-11-19: with a bot account's token the API returns
  ``resolution: null`` and empty aggregations on every resolved question
  tried (MiniBench, FutureEval, Spring 2026), so those questions are
  reported as *hidden*, not resolved.
- **ForecastBench**, for a backtest ledger: the resolution set's 0 or 1,
  and the market's price at the freeze date as a ``market`` reference row
  (the forecaster never saw it).
- **Manual**: a resolution read off the question page by a person, with the
  evidence named (``--manual POST VALUE --evidence URL``).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from augur.clock import parse_iso
from augur.errors import AugurError, MetaculusError
from augur.ledger import Ledger
from augur.metaculus import MetaculusClient
from augur.questions import Question, question_from_post
from augur.score import ABOVE, BELOW

TERMINAL = {"annulled": "annulled", "ambiguous": "ambiguous"}


@dataclass
class ResolveReport:
    checked: int = 0
    resolved: int = 0
    annulled: int = 0
    hidden: int = 0
    open: int = 0
    unchanged: int = 0
    failed: int = 0
    references: int = 0
    scores: int = 0
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        return (
            f"{self.checked} checked: {self.resolved} resolved, {self.annulled} annulled, "
            f"{self.hidden} resolved but hidden from this account, {self.open} not resolved "
            f"yet, {self.unchanged} unchanged, {self.failed} failed; "
            f"{self.references} reference forecasts and {self.scores} Metaculus scores recorded"
        )


def _ledger_questions(conn: sqlite3.Connection, live: bool) -> list[tuple[int, int, str, str]]:
    """``(question_id, post_id, type, payload_json)``: one per question, latest payload."""
    modes = ("live", "dry_run") if live else ("backtest",)
    marks = ", ".join("?" for _ in modes)
    rows = conn.execute(
        f"SELECT q.question_id, q.post_id, q.type, q.payload_json FROM questions q "  # noqa: S608
        f"JOIN runs r ON r.run_id = q.run_id WHERE r.mode IN ({marks}) "
        f"ORDER BY q.recorded_at",
        modes,
    ).fetchall()
    latest: dict[int, tuple[int, int, str, str]] = {}
    for qid, post_id, qtype, payload in rows:
        latest[int(qid)] = (int(qid), int(post_id), str(qtype), str(payload))
    return list(latest.values())


def resolution_value(q: Question, raw: object) -> tuple[str, str | None] | None:
    """``(status, value)`` from the API's ``resolution``; ``None`` when not resolved."""
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in TERMINAL:
        return TERMINAL[lowered], None
    if q.type == "binary":
        if lowered not in ("yes", "no"):
            raise AugurError(f"binary question {q.question_id} resolved to {text!r}")
        return "resolved", lowered
    if q.type == "multiple_choice":
        if text not in (q.options or []):
            raise AugurError(f"question {q.question_id} resolved to {text!r}, not an option")
        return "resolved", text
    if q.type in ("numeric", "discrete"):
        if lowered in (BELOW, ABOVE):
            return "resolved", lowered
        try:
            float(text)
        except ValueError:
            raise AugurError(f"question {q.question_id} resolved to {text!r}") from None
        return "resolved", text
    raise AugurError(f"{q.type} questions are not scored")


def community_values(q: Question, latest: object) -> dict[str, Any] | None:
    """The aggregate's ``latest`` entry in the shape the ledger stores forecasts."""
    if not isinstance(latest, dict):
        return None
    values = latest.get("forecast_values")
    if not isinstance(values, list) or not values:
        return None
    numbers = [float(v) for v in values if isinstance(v, int | float)]
    if len(numbers) != len(values):
        return None
    if q.type == "binary" and len(numbers) == 2:
        return {"probability": numbers[1]}
    if q.type == "multiple_choice" and q.options and len(numbers) == len(q.options):
        return {"probabilities": dict(zip(q.options, numbers, strict=True))}
    if (
        q.type in ("numeric", "discrete")
        and q.scaling is not None
        and len(numbers) == q.scaling.inbound_outcome_count + 1
    ):
        return {"cdf": numbers}
    return None


async def resolve_metaculus(
    ledger: Ledger,
    client: MetaculusClient,
    *,
    out: Callable[[str], None] = print,
) -> ResolveReport:
    report = ResolveReport()
    for qid, post_id, _qtype, _payload in _ledger_questions(ledger.connection, live=True):
        report.checked += 1
        try:
            post = await client.get_post(post_id)
            q = question_from_post(post)
        except (MetaculusError, ValueError) as exc:
            report.failed += 1
            out(f"  {post_id}: could not read the post ({exc})")
            continue
        found_q = post.get("question")
        raw_q: dict[str, Any] = found_q if isinstance(found_q, dict) else {}
        status = str(raw_q.get("status") or post.get("status") or "")
        try:
            found = resolution_value(q, raw_q.get("resolution"))
        except AugurError as exc:
            report.failed += 1
            out(f"  {post_id}: {exc}")
            continue
        _references(ledger, q, raw_q, report)
        if found is None:
            if status == "resolved":
                report.hidden += 1
                out(f"  {post_id}: resolved, but the API hides the resolution from this account")
            else:
                report.open += 1
            continue
        if ledger.latest_resolution(qid) == found:
            report.unchanged += 1
            continue
        state, value = found
        resolved_at = raw_q.get("actual_resolve_time")
        ledger.record_resolution(
            question_id=qid,
            post_id=post_id,
            question_type=q.type,
            status=state,
            value=value,
            resolved_at=parse_iso(resolved_at) if isinstance(resolved_at, str) else None,
            source="metaculus_api",
            evidence=f"GET /api/posts/{post_id}/ question.resolution = {raw_q.get('resolution')!r}",
        )
        if state == "resolved":
            report.resolved += 1
        else:
            report.annulled += 1
        out(f"  {post_id}: {state} {value or ''}".rstrip())
    return report


def _references(ledger: Ledger, q: Question, raw_q: dict[str, Any], report: ResolveReport) -> None:
    """Keep the community aggregate and Metaculus's own scores, where exposed."""
    aggregations = raw_q.get("aggregations")
    if isinstance(aggregations, dict):
        for method in ("recency_weighted", "unweighted"):
            agg = aggregations.get(method)
            if not isinstance(agg, dict):
                continue
            latest = agg.get("latest")
            values = community_values(q, latest)
            if values is not None and isinstance(latest, dict):
                start = latest.get("start_time")
                as_of = (
                    datetime.fromtimestamp(float(start), tz=UTC)
                    if isinstance(start, int | float)
                    else parse_iso(start)
                    if isinstance(start, str)
                    else q.scheduled_close_time
                )
                ledger.record_reference(
                    question_id=q.question_id,
                    post_id=q.post_id,
                    question_type=q.type,
                    forecaster="community",
                    detail=f"{method} aggregate, latest",
                    values=values,
                    as_of=as_of,
                    source=f"GET /api/posts/{q.post_id}/ aggregations.{method}.latest",
                )
                report.references += 1
            for name, score in _scores(agg.get("score_data")):
                ledger.record_metaculus_score(
                    question_id=q.question_id,
                    post_id=q.post_id,
                    forecaster="community",
                    score_type=f"{method}:{name}",
                    score=score,
                )
                report.scores += 1
            break
    mine = raw_q.get("my_forecasts")
    if isinstance(mine, dict):
        for name, score in _scores(mine.get("score_data")):
            ledger.record_metaculus_score(
                question_id=q.question_id,
                post_id=q.post_id,
                forecaster="augur",
                score_type=name,
                score=score,
            )
            report.scores += 1


def _scores(data: object) -> list[tuple[str, float]]:
    if not isinstance(data, dict):
        return []
    return sorted(
        (str(k), float(v))
        for k, v in data.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and k.endswith("score")
    )


def resolve_forecastbench(ledger: Ledger, resolution_set: Path) -> ResolveReport:
    """Resolve a backtest ledger's ForecastBench questions from a resolution set."""
    data = json.loads(resolution_set.read_text(encoding="utf-8"))
    outcomes = {
        (str(r.get("source")), str(r.get("id"))): r
        for r in data.get("resolutions", [])
        if isinstance(r.get("id"), str)
    }
    report = ResolveReport()
    for qid, post_id, qtype, payload_json in _ledger_questions(ledger.connection, live=False):
        payload = json.loads(payload_json)
        key = (str(payload.get("source")), str(payload.get("fb_id")))
        if "fb_id" not in payload:
            continue
        report.checked += 1
        r = outcomes.get(key)
        if r is None or not r.get("resolved") or r.get("resolved_to") not in (0, 1, 0.0, 1.0):
            report.open += 1
            continue
        found = ("resolved", "yes" if float(r["resolved_to"]) == 1.0 else "no")
        if ledger.latest_resolution(qid) == found:
            report.unchanged += 1
            continue
        ledger.record_resolution(
            question_id=qid,
            post_id=post_id,
            question_type=qtype,
            status=found[0],
            value=found[1],
            resolved_at=datetime.combine(
                date.fromisoformat(str(r["resolution_date"])), time(0), tzinfo=UTC
            ),
            source="forecastbench",
            evidence=f"{data.get('question_set')} resolution set: {key[0]} {key[1]} "
            f"resolved_to {r['resolved_to']} on {r['resolution_date']}",
        )
        report.resolved += 1
        freeze_value = payload.get("freeze_value")
        if isinstance(freeze_value, int | float):
            ledger.record_reference(
                question_id=qid,
                post_id=post_id,
                question_type=qtype,
                forecaster="market",
                detail=f"{key[0]} price at the freeze date (not shown to the forecaster)",
                values={"probability": float(freeze_value)},
                as_of=parse_iso(str(payload["freeze_datetime"])),
                source=f"ForecastBench {data.get('question_set')} freeze_datetime_value",
            )
            report.references += 1
    return report


def resolve_manual(
    ledger: Ledger, *, post_id: int, value: str, evidence: str
) -> tuple[str, str | None]:
    """Record a resolution read by a person; the question must be in the ledger."""
    row = ledger.connection.execute(
        "SELECT question_id, type, payload_json FROM questions WHERE post_id = ? "
        "ORDER BY recorded_at DESC LIMIT 1",
        (post_id,),
    ).fetchone()
    if row is None:
        raise AugurError(f"post {post_id} is not in this ledger; resolve only questions it holds")
    if not evidence.strip():
        raise AugurError("a manual resolution needs --evidence (the page or source it came from)")
    q = question_from_post(json.loads(row[2]))
    found = resolution_value(q, value)
    if found is None:
        raise AugurError("give the value the question resolved to, or 'annulled'")
    ledger.record_resolution(
        question_id=int(row[0]),
        post_id=post_id,
        question_type=str(row[1]),
        status=found[0],
        value=found[1],
        resolved_at=None,
        source="manual",
        evidence=evidence,
    )
    return found
