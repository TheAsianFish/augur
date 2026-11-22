"""More draws, or another model, on research an arm already recorded.

Head count and model mix are questions about the forecasters, not the
research, so M7 holds the research fixed: each question's brief and cited
sources are read back from the arm's ledger, exactly as its forecasters saw
them, and new members forecast from them. The rebuilt request is checked
against the recorded one by hash before anything is called, so a new draw
differs from the recorded ones only in its seed (or its model).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from augur.agents.forecaster import forecaster_request
from augur.agents.researcher import ResearchResult
from augur.clock import parse_iso
from augur.errors import AugurError
from augur.forecastbench import FBQuestion
from augur.research.source import DateEvidence, Source

BRIEF_MARK = "\n\nResearch brief:\n"


@dataclass(frozen=True)
class Recorded:
    """One question's research as the recorded member 0 saw it."""

    research: ResearchResult
    request: dict[str, object]
    request_hash: str


def recorded_research(path: Path, tournament: str) -> dict[int, Recorded]:
    """Each question's brief and cited sources from the arm ``tournament`` in ``path``.

    Read from the first successful member-0 forecaster call per question: its
    request holds the brief, its ``call_sources`` rows the sources it cited
    (which the gate admitted, or the ledger would have refused the row).
    """
    out: dict[int, Recorded] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        calls = conn.execute(
            "SELECT c.call_id, c.question_id, c.request_json, c.request_hash, c.run_id "
            "FROM llm_calls c JOIN runs r ON r.run_id = c.run_id "
            "WHERE r.tournament = ? AND c.role = 'forecaster' AND c.member = 0 "
            "AND c.error IS NULL ORDER BY c.started_at",
            (tournament,),
        ).fetchall()
        for call_id, qid, request_json, request_hash, run_id in calls:
            if int(qid) in out:
                continue
            request = json.loads(request_json)
            user = request["messages"][-1]["content"]
            if BRIEF_MARK not in user:
                raise AugurError(f"call {call_id} has no research brief; is {tournament} an arm?")
            brief = user.split(BRIEF_MARK, 1)[1]
            rows = conn.execute(
                "SELECT s.source_id, s.searcher, s.query, s.url, s.title, s.excerpt, "
                "s.content_sha256, s.published_at, s.retrieved_at, s.as_of, s.date_evidence, "
                "s.evidence_detail FROM call_sources cs "
                "JOIN sources s ON s.source_id = cs.source_id "
                "JOIN source_uses u ON u.source_id = s.source_id AND u.run_id = ? "
                "AND u.question_id = ? AND u.admitted = 1 "
                "WHERE cs.call_id = ? GROUP BY s.source_id ORDER BY MIN(u.use_id)",
                (run_id, qid, call_id),
            ).fetchall()
            sources = [
                Source(
                    source_id=r[0],
                    searcher=r[1],
                    query=r[2],
                    url=r[3],
                    title=r[4],
                    excerpt=r[5],
                    content_sha256=r[6],
                    published_at=parse_iso(r[7]) if r[7] else None,
                    retrieved_at=parse_iso(r[8]),
                    as_of=parse_iso(r[9]),
                    date_evidence=DateEvidence(r[10]),
                    evidence_detail=r[11],
                )
                for r in rows
            ]
            research = ResearchResult(queries=[], sources=sources, brief=brief)
            out[int(qid)] = Recorded(research, request, str(request_hash))
    return out


def check_rebuilt(fb: FBQuestion, recorded: Recorded) -> None:
    """Refuse to go on unless the rebuilt member-0 request hashes to the recorded one."""
    q = fb.to_question()
    rebuilt = forecaster_request(q, fb.freeze_datetime, recorded.research, member=0)
    rebuilt = rebuilt.model_copy(
        update={
            "model": recorded.request["model"],
            "reasoning_effort": recorded.request.get("reasoning_effort"),
        }
    )
    if rebuilt.request_hash() != recorded.request_hash:
        raise AugurError(
            f"{fb.source}:{fb.fb_id}: the rebuilt forecaster request does not match the "
            "recorded one, so new draws would not see what the recorded ones saw; was the "
            "prompt changed since that arm ran?"
        )


def research_from(
    recorded: dict[int, Recorded],
) -> Callable[[FBQuestion], ResearchResult | None]:
    """A ``research_for`` for :func:`augur.backtest.run_forecastbench` (checked first)."""

    def research(fb: FBQuestion) -> ResearchResult | None:
        found = recorded.get(fb.question_id)
        if found is None:
            return None
        check_rebuilt(fb, found)
        return found.research

    return research
