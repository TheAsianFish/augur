"""The leakage judge: how much each admitted source gives away, on a 0 to 4 scale.

Q2 asks how much of a backtest's accuracy is leakage. The date gate proves
when a source was written, not what it says: a Wikipedia revision saved
before the cutoff can still expand a template at today's version, and a
page captured before the cutoff can describe a schedule that settles the
question. So a judge reads every source the forecasters were shown, with the
question, the cutoff and the resolution, and rates it on the rubric of El
Lahib et al. (ACL 2026, arXiv 2602.00758, Appendix C.2), whose scores 3 and
4 they call major leakage:

0. noise: no post-cutoff information, or none bearing on the question;
1. topical: post-cutoff information on the topic that would not change a
   forecast;
2. weak signal: a vague or minor directional signal toward the resolution;
3. major or partial direct signal: decisive data for part of the question;
4. full, direct leakage: states the resolution or lets it be computed.

Signals from absence ("no report of X") are capped at 3, as in the paper.

The judge sees the outcome by design and never forecasts, so its calls are
exempt from the model release gate; they are recorded in a run of their own
(``judge:`` before the arm's tournament), in the arm's mode and with the
arm's cutoff, and every source it reads passes the gate again for that run.
The judge is checked against hand labels (:func:`agreement`) before any of
its rates are reported.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from augur.agents.calls import Budget, CallContext, CallRecorder
from augur.clock import Clock, parse_iso, to_iso
from augur.config import Settings
from augur.errors import AugurError, ProviderError
from augur.forecastbench import FBQuestion
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.llm.base import LlmRequest, Message
from augur.llm.router import Router
from augur.log import get_logger
from augur.research.source import DateEvidence, Source

log = get_logger(__name__)

JUDGE_PREFIX = "judge:"
JUDGE_MAX_TOKENS = 4000
MAJOR = 3
"""Scores at or above this are "major leakage", as in the paper."""

JUDGE_SYSTEM = """\
You audit a forecasting backtest for information leakage. A forecaster was asked \
the question below as of the information cutoff date, and was shown the numbered \
sources. You are told how the question actually resolved. Post-cutoff information \
is any event, data point or entity that did not exist or was not public knowledge \
before the cutoff date. Rate each source on this scale:

0 (noise): no post-cutoff information, or information irrelevant to the question \
that would not change a forecaster's probability at all.
1 (topical): post-cutoff information related to the topic that would not help or \
change the forecaster's answer or reasoning.
2 (weak signal): relevant post-cutoff information with a vague or minor \
directional signal toward the question's resolution, such as a vague trend.
3 (major or partial direct signal): post-cutoff information that is a major \
component of, or decisive for part of, the question's resolution.
4 (full direct leakage): post-cutoff information that directly reveals the \
resolved answer, explicitly or by letting it be calculated.

A signal that rests only on the absence of something is capped at 3. Information \
from before the cutoff, however useful, is 0. Reply with a JSON array and nothing \
else, one object per source in order: \
[{"source": "S1", "score": 0, "reason": "at most 25 words"}, ...]"""


@dataclass(frozen=True)
class Judgment:
    source_id: str
    score: int
    reason: str


def judge_request(fb: FBQuestion, sources: Sequence[Source]) -> LlmRequest:
    """The judge's request. The information cutoff is the question's open
    (freeze) instant in every arm: leakage is measured against when the
    question was asked, not against when the leaky arm happened to run."""
    cutoff = fb.freeze_datetime
    blocks = []
    for i, s in enumerate(sources, start=1):
        dated = to_iso(s.published_at)[:10] if s.published_at else "undated"
        blocks.append(f"[S{i}] {s.title} ({s.searcher}, dated {dated}, {s.url})\n{s.excerpt}")
    user = (
        f"Question: {fb.question}\n\nResolution criteria:\n{fb.resolution_criteria}\n\n"
        f"Information cutoff date: {cutoff:%Y-%m-%d %H:%M} UTC\n"
        f"How it resolved: {'Yes' if fb.outcome == 1 else 'No'} "
        f"(resolution date {fb.resolution_date.isoformat()})\n\nSources:\n\n" + "\n\n".join(blocks)
    )
    return LlmRequest(
        model="",  # the router sets each route's model
        messages=(Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=user)),
        temperature=0.0,
        seed=0,
        max_tokens=JUDGE_MAX_TOKENS,
    )


def parse_scores(text: str, n: int) -> dict[int, tuple[int, str]]:
    """``{source index (1-based): (score, reason)}`` from a judge reply.

    Tolerates prose around the array and a missing ``source`` label (the
    position is used). A score outside 0 to 4 or not a number is dropped,
    never guessed.
    """
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    found: dict[int, tuple[int, str]] = {}
    for pos, item in enumerate(raw if isinstance(raw, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("source", "")).strip().upper().lstrip("[").rstrip("]")
        index = int(label[1:]) if re.fullmatch(r"S\d+", label) else pos
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, int | float | str):
            continue
        try:
            value = int(float(score))
        except ValueError:
            continue
        if 0 <= value <= 4 and float(score) == value and 1 <= index <= n:
            found[index] = (value, str(item.get("reason", ""))[:300])
    return found


def load_source(conn: sqlite3.Connection, source_id: str) -> Source:
    row = conn.execute(
        "SELECT source_id, searcher, query, url, title, excerpt, content_sha256, published_at, "
        "retrieved_at, as_of, date_evidence, evidence_detail FROM sources WHERE source_id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise AugurError(f"source {source_id} is not in the ledger; was it written by this run?")
    return Source(
        source_id=row[0],
        searcher=row[1],
        query=row[2],
        url=row[3],
        title=row[4],
        excerpt=row[5],
        content_sha256=row[6],
        published_at=parse_iso(row[7]) if row[7] else None,
        retrieved_at=parse_iso(row[8]),
        as_of=parse_iso(row[9]),
        date_evidence=DateEvidence(row[10]),
        evidence_detail=row[11],
    )


@dataclass(frozen=True)
class Shown:
    """The sources one arm's forecasters were shown for one question, in citation order."""

    run_id: str
    mode: RunMode
    question_id: int
    cutoff: datetime
    source_ids: list[str]


def shown_sources(conn: sqlite3.Connection, tournament: str) -> list[Shown]:
    """For each question of an arm: the sources its forecaster calls cited.

    Read from ``call_sources`` of the member calls, so it is exactly what the
    forecasters had, in the order the brief numbered them (the latest run
    that forecast the question wins, as in scoring).
    """
    rows = conn.execute(
        "SELECT r.run_id, r.mode, q.question_id, q.cutoff, c.call_id FROM forecasts f "
        "JOIN runs r ON r.run_id = f.run_id "
        "JOIN questions q ON q.run_id = f.run_id AND q.question_id = f.question_id "
        "JOIN llm_calls c ON c.call_id = f.call_id "
        "WHERE r.tournament = ? AND f.kind = 'member' ORDER BY f.recorded_at, f.member",
        (tournament,),
    ).fetchall()
    latest: dict[int, Shown] = {}
    for run_id, mode, qid, cutoff, call_id in rows:
        current = latest.get(int(qid))
        if current is not None and current.run_id == run_id:
            continue
        ids = [
            sid
            for (sid,) in conn.execute(
                "SELECT cs.source_id FROM call_sources cs WHERE cs.call_id = ? ORDER BY rowid",
                (call_id,),
            )
        ]
        latest[int(qid)] = Shown(run_id, RunMode(mode), int(qid), parse_iso(cutoff), ids)
    return [s for s in latest.values() if s.source_ids]


@dataclass
class JudgeResult:
    run_id: str
    questions: int = 0
    judged: int = 0
    resumed: int = 0
    failed: list[str] = field(default_factory=list)


def judged_questions(conn: sqlite3.Connection, tournament: str) -> set[int]:
    rows = conn.execute(
        "SELECT DISTINCT j.question_id FROM leak_judgments j JOIN runs r ON r.run_id = j.run_id "
        "WHERE r.tournament = ?",
        (JUDGE_PREFIX + tournament,),
    )
    return {int(q) for (q,) in rows}


async def judge_arm(
    *,
    ledger: Ledger,
    tournament: str,
    questions: Sequence[FBQuestion],
    settings: Settings,
    router: Router,
    clock: Clock,
    code_version: str,
) -> JudgeResult:
    """Rate every source each question of ``tournament`` showed its forecasters.

    One judge call per question rates all of its sources. Resumable: a
    question already judged in this ledger is skipped.
    """
    conn = ledger.connection
    shown = shown_sources(conn, tournament)
    if not shown:
        raise AugurError(f"no forecast in {tournament!r} cited a source; nothing to judge")
    modes = {s.mode for s in shown}
    if len(modes) != 1:
        raise AugurError(f"{tournament!r} mixes run modes {sorted(modes)}; judge one arm at a time")
    mode = modes.pop()
    by_id = {q.question_id: q for q in questions}
    done = judged_questions(conn, tournament)
    judge = router.chain("research")[0]
    run_id = ledger.start_run(
        mode=mode,
        tournament=JUDGE_PREFIX + tournament,
        provider=router.provider(judge).name,
        model=judge.model,
        settings=settings.public_dict() | {"judges": tournament, "rubric": "El Lahib et al. 0-4"},
        code_version=code_version,
    )
    calls = CallRecorder(
        router, ledger, clock, Budget(settings.max_cost_per_question, settings.max_cost_per_run)
    )
    result = JudgeResult(run_id)
    for item in shown:
        fb = by_id.get(item.question_id)
        if fb is None:
            continue
        result.questions += 1
        if item.question_id in done:
            result.resumed += 1
            continue
        q = fb.to_question()
        ledger.record_question(
            run_id, q, item.cutoff, fb.model_dump(mode="json", exclude={"outcome"})
        )
        sources = [load_source(conn, sid) for sid in item.source_ids]
        for s in sources:
            ledger.record_admission(run_id, q.question_id, admit_source(s, item.cutoff, mode))
        ctx = CallContext(run_id=run_id, question_id=q.question_id, role="judge")
        try:
            done_call = await calls.complete(
                ctx, judge_request(fb, sources), [s.source_id for s in sources]
            )
        except (ProviderError, AugurError) as exc:
            result.failed.append(f"{fb.fb_id}: {exc}")
            log.warning("judge call failed", question=fb.fb_id, error=str(exc))
            continue
        scores = parse_scores(done_call.response.text, len(sources))
        if not scores:
            result.failed.append(f"{fb.fb_id}: the judge's reply had no usable scores")
            continue
        for index, (score, reason) in sorted(scores.items()):
            ledger.record_judgment(
                run_id=run_id,
                question_id=q.question_id,
                source_id=sources[index - 1].source_id,
                call_id=done_call.call_id,
                judge_model=done_call.route.model,
                score=score,
                rationale=reason,
            )
        result.judged += 1
    return result


def judgments(path: Path, tournament: str) -> dict[tuple[int, str], int]:
    """``{(question id, source id): score}`` for one arm (the latest judgment wins)."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT j.question_id, j.source_id, j.score FROM leak_judgments j "
            "JOIN runs r ON r.run_id = j.run_id WHERE r.tournament = ? ORDER BY j.recorded_at",
            (JUDGE_PREFIX + tournament,),
        ).fetchall()
    return {(int(q), str(s)): int(v) for q, s, v in rows}


# -- agreement with hand labels ------------------------------------------------


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 5) -> float:
    """Cohen's kappa with quadratic weights over the scores ``0..k-1``."""
    if len(a) != len(b) or not a:
        raise ValueError("two equal, non-empty lists of scores are needed")
    n = len(a)
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[x][y] += 1.0
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * observed[i][j]
            den += w * row[i] * col[j] / n
    return 1.0 - num / den if den else 1.0


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("two equal, non-empty lists are needed")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


@dataclass(frozen=True)
class Agreement:
    n: int
    exact_merged: float
    """Share of exact agreement with scores 0 and 1 merged, as the paper reports."""
    qwk: float
    major_agree: float
    """Share agreeing on "3 or 4" against "0 to 2"."""
    major_kappa: float
    human_major: int
    judge_major: int


def agreement(pairs: Sequence[tuple[int, int]]) -> Agreement:
    """``pairs`` are ``(hand label, judge score)``."""
    if not pairs:
        raise ValueError("no hand label has a judge score")
    hand = [h for h, _ in pairs]
    judge = [j for _, j in pairs]
    merged = sum(max(h, 1) == max(j, 1) for h, j in pairs) / len(pairs)
    hm = [h >= MAJOR for h in hand]
    jm = [j >= MAJOR for j in judge]
    return Agreement(
        n=len(pairs),
        exact_merged=merged,
        qwk=quadratic_weighted_kappa(hand, judge),
        major_agree=sum(x == y for x, y in zip(hm, jm, strict=True)) / len(pairs),
        major_kappa=cohen_kappa(hm, jm),
        human_major=sum(hm),
        judge_major=sum(jm),
    )
