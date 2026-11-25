"""M13: does dated news move the as-of-open arm? Scored from the ledgers.

Arm G (PLAN.md, ``m13-registration``) is arm O's evidence plus up to three
news sources from the ``gdelt`` searcher, each read from a Wayback capture
at or before the cutoff. Where nothing was added, G's inputs are O's and its
forecast is O's recorded one, so G − O is news and nothing else.

Everything here reads committed ledgers and is deterministic (the bootstrap
is seeded), so ``augur report`` renders it and ``make eval`` checks it.
"""

from __future__ import annotations

import math
import re
import sqlite3
import statistics
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from augur.forecastbench import FBQuestion
from augur.judge import JUDGE_PREFIX, MAJOR
from augur.q2 import ArmRow, ArmSource, CostRow, DiffRow, arm_row, cost_row, diff_row, forecasts
from augur.redraw import recorded_research
from augur.replay import recorded_prices

RECENT = timedelta(days=30)
DIGEST_TITLE = 200
"""Characters: no real headline in arm G is longer; the one digest page's is thousands."""
HINDCAST_GAIN = 0.055
"""Hindcast's Qwen3-32B gain from dated retrieval (0.234 to 0.179), the verdict's bar."""
Z_95, Z_80 = 1.959964, 0.841621
_SEEN = re.compile(r"seendate=(\d{8}T\d{6}Z)")


def seendate(evidence_detail: str) -> datetime | None:
    """The GDELT ``seendate`` a news source's evidence names, or ``None``."""
    match = _SEEN.search(evidence_detail)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class News:
    """One admitted news source of arm G."""

    question_id: int
    source_id: str
    seen: datetime | None
    url: str
    title: str = ""

    @property
    def digest(self) -> bool:
        """A wire-digest page whose GKG "title" is a whole feed (found after the run)."""
        return len(self.title) > DIGEST_TITLE


def news_sources(path: Path, tournament: str) -> list[News]:
    """The news sources the gate admitted for each question in its deciding run, once each.

    A question's deciding run is the one that recorded its latest outcome: a
    question whose draws failed and whose retry gained nothing keeps O's
    forecast, so the first attempt's news does not count for it.
    """
    out: dict[tuple[int, str], News] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        deciding: dict[int, str] = {}
        for qid, run_id in conn.execute(
            "SELECT o.question_id, o.run_id FROM question_outcomes o "
            "JOIN runs r ON r.run_id = o.run_id WHERE r.tournament = ? ORDER BY o.rowid",
            (tournament,),
        ):
            deciding[int(qid)] = str(run_id)
        for qid, run_id, sid, detail, url, title in conn.execute(
            "SELECT u.question_id, u.run_id, s.source_id, s.evidence_detail, s.url, s.title "
            "FROM source_uses u JOIN runs r ON r.run_id = u.run_id "
            "JOIN sources s ON s.source_id = u.source_id "
            "WHERE r.tournament = ? AND u.admitted = 1 AND s.searcher = 'gdelt' "
            "ORDER BY u.use_id",
            (tournament,),
        ):
            key = (int(qid), str(sid))
            if deciding.get(int(qid)) == str(run_id) and key not in out:
                out[key] = News(int(qid), str(sid), seendate(str(detail)), str(url), str(title))
    return list(out.values())


def news_judgments(path: Path, tournament: str, news: set[str]) -> list[int]:
    """The latest judge score of each judged news source (other sources left out)."""
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT j.question_id, j.source_id, j.score FROM leak_judgments j "
            "JOIN runs r ON r.run_id = j.run_id WHERE r.tournament = ? ORDER BY j.recorded_at",
            (JUDGE_PREFIX + tournament,),
        ).fetchall()
    latest = {(int(q), str(s)): int(v) for q, s, v in rows if str(s) in news}
    return list(latest.values())


def n_needed(sd: float, effect: float) -> int | None:
    """Questions to detect ``effect`` at 80% power, two-sided 5% (M7's normal rule)."""
    if effect <= 0 or sd <= 0:
        return None
    return math.ceil(((Z_95 + Z_80) * sd / effect) ** 2)


def verdict(diff: DiffRow) -> str:
    """Registration item 5, applied to G − O over every question."""
    low, high = diff.brier.low, diff.brier.high
    if high < 0:
        return "moved"
    if low > 0 or low > -HINDCAST_GAIN:
        return "not the limit"
    return "unresolved"


@dataclass(frozen=True)
class Q13:
    sample: int
    gained: int
    gained_recent: int
    news: int
    pages: int
    base_per_question: float
    g_per_question: float
    arms: list[ArmRow]
    diffs: list[DiffRow]
    priced: int
    priced_diffs: list[DiffRow]
    unpriced: int
    unpriced_diffs: list[DiffRow]
    gained_diffs: list[DiffRow]
    digest_only: int
    real_uses: int
    digest_questions: int
    article_diffs: list[DiffRow]
    """Exploratory, chosen after the run: G − O where a real article (not the digest) came."""
    sd: float
    needed: dict[float, int | None]
    verdict: str
    judged: int
    judged_major: int
    judge_scores: tuple[int, int, int, int, int]
    cost: CostRow | None
    domains: list[tuple[str, int]]


def score_q13(
    *,
    questions: Sequence[FBQuestion],
    sample: int,
    n_arm: ArmSource,
    o_arm: ArmSource,
    g_arm: ArmSource,
    prices: ArmSource,
) -> Q13 | None:
    """M13's tables on C2's fixed sample, or ``None`` before arm G has run."""
    if not g_arm.path.exists():
        return None
    ordered = sorted(questions, key=lambda q: q.question_id)[:sample]
    outcome = {q.question_id: q.outcome for q in ordered}
    opened = {q.question_id: q.freeze_datetime for q in ordered}
    n_f = forecasts(n_arm.path, n_arm.tournament)
    o_f = forecasts(o_arm.path, o_arm.tournament)
    g_own = forecasts(g_arm.path, g_arm.tournament)
    news = [n for n in news_sources(g_arm.path, g_arm.tournament) if n.question_id in outcome]
    gained_q = {n.question_id for n in news}
    # Where nothing was added G's inputs are O's: its forecast is O's (item 3).
    g_f = {q: g_own[q] if q in gained_q and q in g_own else o_f[q] for q in o_f}
    common = [q.question_id for q in ordered if all(q.question_id in f for f in (n_f, o_f, g_f))]
    if not common:
        return None
    missing = [q for q in common if q in gained_q and q not in g_own]
    if missing:
        # News was admitted but no forecast made from it: the arm is unfinished.
        return None
    ys = [outcome[q] for q in common]
    col = {
        "N": [n_f[q] for q in common],
        "O": [o_f[q] for q in common],
        "G": [g_f[q] for q in common],
    }
    names = {
        "N": "N: no retrieval",
        "O": "O: one pass, sources as of open",
        "G": "G: O's sources plus dated news",
    }
    arms = [arm_row(names[a], col[a], ys) for a in ("N", "O", "G")]
    go = diff_row("G − O (what dated news adds)", col["G"], col["O"], ys)
    diffs = [
        go,
        diff_row("G − N (news and the rest of retrieval)", col["G"], col["N"], ys),
        diff_row("O − N (M6, for reference)", col["O"], col["N"], ys),
    ]
    quotes = {
        q: s.quote.probability
        for q, s in recorded_prices(prices.path, prices.tournament).items()
        if s.quote is not None
    }

    def split(qs: list[int], tag: str) -> list[DiffRow]:
        if not qs:
            return []
        y = [outcome[q] for q in qs]
        return [
            diff_row(f"G − O ({tag})", [g_f[q] for q in qs], [o_f[q] for q in qs], y),
            diff_row(f"G − N ({tag})", [g_f[q] for q in qs], [n_f[q] for q in qs], y),
        ]

    priced_q = [q for q in common if q in quotes]
    unpriced_q = [q for q in common if q not in quotes]
    gained_list = [q for q in common if q in gained_q]
    gained_diffs = []
    if gained_list:
        y = [outcome[q] for q in gained_list]
        gained_diffs.append(
            diff_row(
                f"G − O (questions that gained news, n = {len(gained_list)})",
                [g_f[q] for q in gained_list],
                [o_f[q] for q in gained_list],
                y,
            )
        )
    real = {n.question_id for n in news if not n.digest}
    with_digest = {n.question_id for n in news if n.digest}
    article_q = [q for q in common if q in real]
    article_diffs = []
    if article_q:
        y = [outcome[q] for q in article_q]
        article_diffs.append(
            diff_row(
                f"G − O (gained a real article, n = {len(article_q)}; exploratory)",
                [g_f[q] for q in article_q],
                [o_f[q] for q in article_q],
                y,
            )
        )
    recent = {
        n.question_id
        for n in news
        if n.seen is not None and opened[n.question_id] - RECENT <= n.seen <= opened[n.question_id]
    }
    per_question = {q: 0 for q in common}
    for n in news:
        if n.question_id in per_question:
            per_question[n.question_id] += 1
    base = recorded_research(o_arm.path, o_arm.tournament)
    base_counts = [len(base[q].research.sources) for q in common if q in base]
    base_mean = sum(base_counts) / len(base_counts) if base_counts else 0.0
    per_q = [(g - y) ** 2 - (o - y) ** 2 for g, o, y in zip(col["G"], col["O"], ys, strict=True)]
    sd = statistics.stdev(per_q) if len(per_q) > 1 else 0.0
    news_ids = {n.source_id for n in news}
    scores = news_judgments(g_arm.path, g_arm.tournament, news_ids)
    counts = [0] * 5
    for v in scores:
        counts[v] += 1
    domains: dict[str, int] = {}
    for n in news:
        host = re.sub(r"^https?://web\.archive\.org/web/\d+/", "", n.url)
        host = re.sub(r"^https?://", "", host).split("/", 1)[0].removeprefix("www.")
        domains[host] = domains.get(host, 0) + 1
    return Q13(
        sample=len(common),
        gained=len(gained_list),
        gained_recent=len(recent & set(common)),
        news=sum(per_question.values()),
        pages=len({n.source_id for n in news if n.question_id in per_question}),
        base_per_question=base_mean,
        g_per_question=base_mean + sum(per_question.values()) / len(common),
        arms=arms,
        diffs=diffs,
        priced=len(priced_q),
        priced_diffs=split(priced_q, f"priced, n = {len(priced_q)}"),
        unpriced=len(unpriced_q),
        unpriced_diffs=split(unpriced_q, f"unpriced, n = {len(unpriced_q)}"),
        gained_diffs=gained_diffs,
        digest_only=len([q for q in gained_list if q not in real]),
        real_uses=sum(1 for n in news if not n.digest and n.question_id in per_question),
        digest_questions=len(with_digest & set(common)),
        article_diffs=article_diffs,
        sd=sd,
        needed={e: n_needed(sd, e) for e in (0.02, HINDCAST_GAIN)},
        verdict=verdict(go),
        judged=len(scores),
        judged_major=sum(1 for v in scores if v >= MAJOR),
        judge_scores=(counts[0], counts[1], counts[2], counts[3], counts[4]),
        cost=cost_row("G", g_arm.path, g_arm.tournament, set(common)),
        domains=sorted(domains.items(), key=lambda kv: (-kv[1], kv[0]))[:8],
    )
