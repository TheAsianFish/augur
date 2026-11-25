"""``augur standings``: where Augur ranks in a MiniBench round, from Metaculus's leaderboard.

A round's primary leaderboard lists every forecaster with the SUM of its spot
peer scores over the questions it covered (``score``), a confidence interval on
that sum (``ci_lower``, ``ci_upper``) and the number of questions covered
(``coverage``, possibly fractional). Bots are not marked as template bots,
open-source bots or house bots; those groups are recovered from names here.

Average spot peer score per question is ``score / coverage``. Its interval is
Metaculus's interval on the sum divided by ``coverage``
(``ci_lower / coverage .. ci_upper / coverage``). That is exact if Metaculus's
interval is a bootstrap of the mean scaled by coverage; we did not verify their
method, so read the interval as theirs, rescaled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from augur.clock import Clock, SystemClock, parse_iso, to_iso
from augur.config import Settings
from augur.errors import ConfigError, MetaculusError
from augur.llm.catalog import Catalog, CatalogEntry
from augur.log import get_logger
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter

_log = get_logger("augur.standings")

BOT_USERNAME_ENV = "AUGUR_BOT_USERNAME"

OPEN_SOURCE_BOTS: Mapping[str, str] = {
    "nostreambot": ("https://www.lesswrong.com/posts/wZBbDqzfBjYG58CxK (Spring 2026 results, MIT)"),
    "Panshul42": "https://www.metaculus.com/notebooks/39140/ (Q2 2025 winner, open source)",
}
"""Open-source bots cannot be identified from the API; this is a sourced, hand-kept list."""

TEMPLATE_NAME = re.compile(r"^metac-.+\+asknews")
"""Metaculus's template bot: ``metac-<model>+asknews``, with an optional experiment suffix."""
TEMPLATE_VARIANT = re.compile(r"^metac-.+\+asknews.+")
AGGREGATE_ORDER = ("unweighted", "recency_weighted", "reputation_weighted")
SNAPSHOT_VERSION = 1


class Model(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class LeaderboardUser(Model):
    id: int
    username: str
    is_bot: bool = False


class Entry(Model):
    user: LeaderboardUser | None = None
    aggregation_method: str | None = None
    score: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    rank: int | None = None
    excluded: bool = False
    exclusion_status: int = 0
    coverage: float = 0.0
    contribution_count: float | None = None

    @property
    def name(self) -> str:
        if self.user is not None:
            return self.user.username
        return f"community prediction ({self.aggregation_method})"

    @property
    def average(self) -> float | None:
        """Spot peer score per question covered: ``score / coverage``."""
        return self.score / self.coverage if self.coverage > 0 else None

    @property
    def average_interval(self) -> tuple[float, float] | None:
        """Metaculus's interval on the sum, divided by coverage (see the module docstring)."""
        if self.coverage <= 0 or self.ci_lower is None or self.ci_upper is None:
            return None
        return self.ci_lower / self.coverage, self.ci_upper / self.coverage


class Leaderboard(Model):
    project_name: str = ""
    project_slug: str = ""
    score_type: str = ""
    finalized: bool = False
    prize_pool: float | None = None
    max_coverage: float = 0.0
    is_primary_leaderboard: bool = False
    entries: list[Entry] = []


class RoundQuestion(Model):
    post_id: int
    question_id: int | None
    title: str
    status: str
    type: str | None = None
    closes_at: str | None = None


class RoundInfo(Model):
    id: int
    slug: str
    name: str = ""
    start_date: str | None = None
    close_date: str | None = None
    forecasting_end_date: str | None = None
    score_type: str = ""


class RoundStandings(BaseModel):
    """Everything a rendering needs, as saved in a snapshot."""

    model_config = ConfigDict(frozen=True)

    round: RoundInfo
    leaderboard_raw: dict[str, Any]
    questions: list[RoundQuestion]
    fetched_at: datetime
    source_url: str
    augur_username: str | None = None

    def board(self) -> Leaderboard:
        return Leaderboard.model_validate(self.leaderboard_raw)


# -- fetching and snapshots ------------------------------------------------------------


def questions_from_posts(posts: Iterable[dict[str, Any]]) -> list[RoundQuestion]:
    out: list[RoundQuestion] = []
    for post in posts:
        post_id = post.get("id")
        if not isinstance(post_id, int):
            continue
        q = post.get("question")
        q = q if isinstance(q, dict) else {}
        qid = q.get("id")
        out.append(
            RoundQuestion(
                post_id=post_id,
                question_id=qid if isinstance(qid, int) else None,
                title=str(post.get("title") or q.get("title") or ""),
                status=str(post.get("status") or ""),
                type=q.get("type") if isinstance(q.get("type"), str) else None,
                closes_at=post.get("scheduled_close_time") or q.get("scheduled_close_time"),
            )
        )
    return sorted(out, key=lambda r: r.post_id)


async def resolve_username(client: MetaculusClient, env: Mapping[str, str]) -> str | None:
    """The account's username: ``GET /users/me/`` first, then ``AUGUR_BOT_USERNAME``."""
    if client.has_token:
        try:
            me = await client.get_me()
        except MetaculusError as exc:
            _log.warning("standings.users_me_failed", reason=str(exc))
        else:
            name = me.get("username")
            if isinstance(name, str) and name:
                return name
    fallback = env.get(BOT_USERNAME_ENV, "").strip()
    return fallback or None


async def fetch_round(
    client: MetaculusClient,
    slug: str,
    clock: Clock | None = None,
    *,
    augur_username: str | None = None,
) -> RoundStandings:
    """Fetch the round's record, its primary leaderboard and its questions."""
    clock = clock or SystemClock()
    info = RoundInfo.model_validate(await client.get_tournament(slug))
    boards = await client.get_leaderboards(info.id)
    primary = [b for b in boards if b.get("is_primary_leaderboard")]
    if not primary:
        raise MetaculusError(
            f"project {info.id} ({slug}) has no primary leaderboard; check the round slug", None
        )
    posts = await client.list_posts(slug)
    return RoundStandings(
        round=info,
        leaderboard_raw=primary[0],
        questions=questions_from_posts(posts),
        fetched_at=clock.now(),
        source_url=f"{client.base_url}/leaderboards/project/{info.id}/",
        augur_username=augur_username,
    )


def snapshot_text(round_: RoundStandings) -> str:
    """Byte-stable JSON: sorted keys, indent 1, one trailing newline."""
    body = {
        "version": SNAPSHOT_VERSION,
        "round": round_.round.model_dump(),
        "leaderboard": round_.leaderboard_raw,
        "questions": [q.model_dump() for q in round_.questions],
        "fetched_at": to_iso(round_.fetched_at),
        "source_url": round_.source_url,
        "augur_username": round_.augur_username,
    }
    return json.dumps(body, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def load_snapshot(path: Path) -> RoundStandings:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("version") != SNAPSHOT_VERSION:
            raise ValueError(f"snapshot version {body.get('version')!r}")
        return RoundStandings(
            round=RoundInfo.model_validate(body["round"]),
            leaderboard_raw=body["leaderboard"],
            questions=[RoundQuestion.model_validate(q) for q in body["questions"]],
            fetched_at=parse_iso(body["fetched_at"]),
            source_url=body["source_url"],
            augur_username=body.get("augur_username"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ConfigError(
            f"cannot read standings snapshot {path} ({exc}); fetch it again with "
            "augur standings --round SLUG --out PATH"
        ) from None


# -- classification and picking --------------------------------------------------------


def classify(entry: Entry, augur_username: str | None) -> str:
    """A short label for what kind of forecaster an entry is."""
    if entry.user is None:
        return "community prediction"
    name = entry.user.username
    if augur_username and name == augur_username:
        return "Augur"
    if TEMPLATE_VARIANT.match(name):
        return "Metaculus template bot (experiment variant)"
    if TEMPLATE_NAME.match(name):
        return "Metaculus template bot"
    if name in OPEN_SOURCE_BOTS:
        return "open-source bot"
    if name.startswith(("metac-", "metaculus-")):
        return "Metaculus house bot (not the template)"
    return "bot" if entry.user.is_bot else "human"


def _pool(entries: Iterable[Entry]) -> list[Entry]:
    return [e for e in entries if e.user is not None and not e.excluded and e.rank is not None]


def _best(entries: Iterable[Entry]) -> Entry | None:
    ranked = _pool(entries)
    if not ranked:
        return None
    return min(ranked, key=lambda e: (e.rank if e.rank is not None else 0, -e.score))


def rank_one(board: Leaderboard) -> Entry | None:
    return _best(board.entries)


def augur_entry(board: Leaderboard, username: str | None) -> Entry | None:
    if not username:
        return None
    for e in board.entries:
        if e.user is not None and e.user.username == username:
            return e
    return None


def best_of_kind(board: Leaderboard, kinds: Sequence[str], username: str | None) -> Entry | None:
    return _best(e for e in board.entries if classify(e, username) in kinds)


def best_template(board: Leaderboard) -> Entry | None:
    return best_of_kind(
        board,
        ("Metaculus template bot", "Metaculus template bot (experiment variant)"),
        None,
    )


def best_house_bot(board: Leaderboard) -> Entry | None:
    return best_of_kind(board, ("Metaculus house bot (not the template)",), None)


def best_open_source(board: Leaderboard) -> Entry | None:
    return best_of_kind(board, ("open-source bot",), None)


def community_aggregates(board: Leaderboard) -> list[Entry]:
    found = {e.aggregation_method: e for e in board.entries if e.user is None}
    ordered = [found[m] for m in AGGREGATE_ORDER if m in found]
    ordered += [
        e for m, e in sorted(found.items(), key=lambda kv: str(kv[0])) if m not in AGGREGATE_ORDER
    ]
    return ordered


# -- missed questions ------------------------------------------------------------------


class Outcome(Model):
    question_id: int
    status: str
    reason: str
    submitted: bool
    recorded_at: str
    run_id: str


def read_outcomes(paths: Sequence[Path]) -> dict[int, list[Outcome]]:
    """``question_outcomes`` rows of the given ledgers (opened read-only), by question id."""
    found: dict[int, list[Outcome]] = {}
    for path in paths:
        if not path.exists():
            raise ConfigError(f"ledger {path} does not exist; pass the path of a ledger file")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT question_id, status, reason, submitted, recorded_at, run_id "
                "FROM question_outcomes"
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []  # a ledger older than schema 4 has no outcomes
        finally:
            conn.close()
        for qid, status, reason, submitted, recorded_at, run_id in rows:
            found.setdefault(int(qid), []).append(
                Outcome(
                    question_id=qid,
                    status=status,
                    reason=reason,
                    submitted=bool(submitted),
                    recorded_at=recorded_at,
                    run_id=run_id,
                )
            )
    return found


def explain(question: RoundQuestion, outcomes: Mapping[int, list[Outcome]]) -> tuple[bool, str]:
    """(forecast on Metaculus?, why not): any submitted forecast counts; otherwise the
    latest outcome row says why, and no row means no run saw the question."""
    rows = outcomes.get(question.question_id, []) if question.question_id is not None else []
    if any(r.submitted and r.status == "forecast" for r in rows):
        return True, "forecast and submitted"
    if not rows:
        return False, "no run saw it"
    latest = max(rows, key=lambda r: (r.recorded_at, r.run_id))
    if latest.status == "forecast":
        return False, "forecast in a dry run, not submitted"
    return False, f"{latest.status}: {latest.reason}"


# -- rendering -------------------------------------------------------------------------


def _fmt_avg(entry: Entry) -> str:
    avg, interval = entry.average, entry.average_interval
    if avg is None:
        return "n/a"
    if interval is None:
        return f"{avg:.2f}"
    return f"{avg:.2f} [{interval[0]:.2f}, {interval[1]:.2f}]"


def _fmt_questions(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _row(label: str, entry: Entry | None, augur_username: str | None) -> list[str] | None:
    if entry is None:
        return None
    rank = str(entry.rank) if entry.rank is not None else "-"
    return [
        label,
        rank,
        entry.name,
        classify(entry, augur_username),
        _fmt_questions(entry.coverage),
        _fmt_avg(entry),
        f"{entry.score:.1f}",
    ]


def _table(rows: list[list[str]]) -> list[str]:
    head = [
        "Row",
        "Rank",
        "Name",
        "Kind",
        "Questions",
        "Avg spot peer score / question [interval]",
        "Summed score",
    ]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in rows]
    return lines


def render(round_: RoundStandings, outcomes: Mapping[int, list[Outcome]] | None = None) -> str:
    """The round's standings as Markdown. ``outcomes`` is None when no ledger was given."""
    board = round_.board()
    user = round_.augur_username
    name = board.project_name or round_.round.name or round_.round.slug
    lines = [
        f"## {name} (`{round_.round.slug}`)",
        "",
        f"Finalized: {'yes' if board.finalized else 'no'}. Leaderboard entries: "
        f"{len(board.entries)}. Most questions any entry covers: "
        f"{_fmt_questions(board.max_coverage)}. "
        f"Score type: {board.score_type or round_.round.score_type}. "
        f"Fetched {to_iso(round_.fetched_at)} from {round_.source_url}.",
        "",
    ]
    if not board.entries:
        lines += [
            f"The leaderboard has no entries yet (round starts {round_.round.start_date}, "
            f"closes {round_.round.close_date}); there is nothing to compare.",
            "",
        ]
    rows: list[list[str]] = []
    mine = augur_entry(board, user)
    if board.entries:
        for label, entry in (
            ("Rank 1", rank_one(board)),
            ("Augur", mine),
            ("Best template bot", best_template(board)),
            ("Best Metaculus house bot (not the template)", best_house_bot(board)),
            ("Best listed open-source bot", best_open_source(board)),
        ):
            row = _row(label, entry, user)
            if row is not None:
                rows.append(row)
        for entry in community_aggregates(board):
            row = _row("Community prediction", entry, user)
            if row is not None:
                rows.append(row)
        lines += _table(rows)
        lines.append("")
    notes: list[str] = []
    if board.entries:
        if not user:
            notes.append(
                f"Augur: account unknown (no token to ask /users/me/ and no {BOT_USERNAME_ENV}); "
                "it is not shown on the leaderboard."
            )
        elif mine is None:
            notes.append(
                f"Augur ({user}) is not on this round's leaderboard: it has no scored "
                "forecast here."
            )
        if best_open_source(board) is None:
            notes.append("Best listed open-source bot: no listed open-source bot in this round.")
        if best_house_bot(board) is None:
            notes.append("Best Metaculus house bot: none on this leaderboard.")
        if best_template(board) is None:
            notes.append("Best template bot: none on this leaderboard.")
        notes.append(
            "Average = summed score / questions covered; the interval is Metaculus's interval on "
            "the sum divided by the same count (their method is not verified here). Rank is "
            "Metaculus's, by summed score; ties share a rank. Template variants "
            "(`+asknews[...]`, `+asknews(...)`) are Metaculus's experiments on the template. "
            "Open-source bots are matched against a short sourced list: "
            + "; ".join(f"{n} ({src})" for n, src in OPEN_SOURCE_BOTS.items())
            + "."
        )
    lines += [f"- {n}" for n in notes]
    if notes:
        lines.append("")
    lines += _render_missed(round_, outcomes)
    return "\n".join(lines).rstrip("\n") + "\n"


def _render_missed(
    round_: RoundStandings, outcomes: Mapping[int, list[Outcome]] | None
) -> list[str]:
    lines = [f"### Questions Augur did not forecast ({round_.round.slug})", ""]
    if outcomes is None:
        return [
            *lines,
            "No ledger given (--ledger PATH), so missed questions are not explained.",
            "",
        ]
    if not round_.questions:
        return [*lines, "The snapshot lists no questions for this round.", ""]
    missed: list[str] = []
    done = 0
    for q in round_.questions:
        forecast, why = explain(q, outcomes)
        if forecast:
            done += 1
        else:
            title = q.title if len(q.title) <= 90 else q.title[:87] + "..."
            qid = f", question {q.question_id}" if q.question_id is not None else ""
            missed.append(f"- post {q.post_id}{qid} [{q.status}] {title}: {why}")
    lines.append(
        f"{done} of {len(round_.questions)} questions forecast and submitted; "
        f"{len(missed)} not forecast."
    )
    if missed:
        lines += ["", *missed]
    return [*lines, ""]


# -- command ---------------------------------------------------------------------------


# -- template bots by model, the agent harness, the proposed floor -----------------------

EFFORT_SUFFIX = re.compile(r"-(?:minimal|low|medium|high|xhigh)$")
PLANNER_INPUT_TOKENS = 1_500
BRIEF_INPUT_TOKENS = 6_000
DRAW_INPUT_TOKENS = 3_000
ROUND_QUESTIONS = 60
_TEMPLATE_PARTS = re.compile(r"^metac-(?!agent-)(?P<model>.+?)\+asknews(?P<variant>.*)$")


@dataclass(frozen=True)
class TemplateRow:
    name: str
    label: str  # the model as the bot's name spells it, e.g. ``claude-opus-5-5-high``
    key: str  # the label without a reasoning-effort suffix: ``claude-opus-5-5``
    variant: str  # text after ``+asknews`` (an experiment), empty for the plain template
    rank: int | None
    entry: Entry

    @property
    def average(self) -> float | None:
        return self.entry.average


def template_parts(name: str) -> tuple[str, str] | None:
    """``metac-X+asknews<variant>`` -> ``(X, variant)``; None for anything else.

    ``metac-agent-*`` are the agent harness, not templates; other ``metac-*``
    bots (autocast, claude-code, codex, fusion, deepnews) have no ``+asknews``.
    """
    match = _TEMPLATE_PARTS.match(name)
    return (match["model"], match["variant"]) if match else None


def agent_model(name: str) -> str | None:
    return name[len("metac-agent-") :] if name.startswith("metac-agent-") else None


def _template_rows(round_: RoundStandings) -> list[TemplateRow]:
    rows: list[TemplateRow] = []
    for e in round_.board().entries:
        if e.user is None or e.average is None:
            continue
        parts = template_parts(e.user.username)
        if parts is None:
            continue
        label, variant = parts
        rows.append(
            TemplateRow(e.user.username, label, EFFORT_SUFFIX.sub("", label), variant, e.rank, e)
        )
    return rows


def template_table(round_: RoundStandings) -> list[TemplateRow]:
    """Every Metaculus template bot, highest average spot peer score first."""
    return sorted(_template_rows(round_), key=lambda r: (-(r.average or 0.0), r.name))


def _plain_by_key(rows: Iterable[TemplateRow]) -> dict[str, TemplateRow]:
    """The plain (no experiment suffix) template per model; the unsuffixed label wins a tie."""
    chosen: dict[str, TemplateRow] = {}
    for r in rows:
        if r.variant:
            continue
        have = chosen.get(r.key)
        if have is None or (r.label == r.key and have.label != have.key):
            chosen[r.key] = r
    return chosen


@dataclass(frozen=True)
class AgentPair:
    model: str
    agent: Entry
    template: TemplateRow

    @property
    def difference(self) -> float:
        assert self.agent.average is not None
        assert self.template.average is not None
        return self.agent.average - self.template.average

    @property
    def ahead(self) -> str:
        """``agent``, ``template`` or ``tie``, by the two average spot peer scores."""
        if self.difference > 0:
            return "agent"
        return "template" if self.difference < 0 else "tie"

    @property
    def intervals_overlap(self) -> bool | None:
        """Whether the two entries' intervals overlap; None if either has none."""
        a, t = self.agent.average_interval, self.template.entry.average_interval
        if a is None or t is None:
            return None
        return a[0] <= t[1] and t[0] <= a[1]


def agent_vs_template(round_: RoundStandings) -> tuple[list[AgentPair], list[str], list[str]]:
    """``(pairs, agents without a template, templates without an agent)``.

    ``metac-agent-X`` matches the plain template whose model, less any effort
    suffix, is ``X``. Pairs are sorted by the difference, largest first.
    """
    plain = _plain_by_key(_template_rows(round_))
    pairs: list[AgentPair] = []
    lone_agents: list[str] = []
    used: set[str] = set()
    for e in round_.board().entries:
        if e.user is None:
            continue
        model = agent_model(e.user.username)
        if model is None or e.average is None:
            continue
        template = plain.get(EFFORT_SUFFIX.sub("", model))
        if template is None:
            lone_agents.append(e.user.username)
        else:
            pairs.append(AgentPair(model, e, template))
            used.add(template.key)
    lone_templates = sorted(r.name for k, r in plain.items() if k not in used)
    pairs.sort(key=lambda p: -p.difference)
    return pairs, sorted(lone_agents), lone_templates


PAIR_MIN_QUESTIONS = 30
"""A same-model pair is shown only when both entries have this many scored questions."""


def shows_agent_pairs(round_: RoundStandings) -> bool:
    """Finalized rounds, and the FutureEval project while it runs (it is read as it scores)."""
    return round_.board().finalized or "futureeval" in round_.round.slug.lower()


def scored_agent_pairs(
    round_: RoundStandings, min_questions: float = PAIR_MIN_QUESTIONS
) -> tuple[list[AgentPair], list[AgentPair]]:
    """``(pairs where both entries have >= min_questions, the pairs left out)``.

    The pairing is ``agent_vs_template``'s; only the coverage threshold is added.
    """
    pairs, _, _ = agent_vs_template(round_)
    kept = [
        p
        for p in pairs
        if p.agent.coverage >= min_questions and p.template.entry.coverage >= min_questions
    ]
    left = [p for p in pairs if p not in kept]
    return kept, left


def agent_pairs_markdown(round_: RoundStandings, min_questions: float = PAIR_MIN_QUESTIONS) -> str:
    """The same-model agent-against-template pairs of one round, with both intervals."""
    board = round_.board()
    kept, left = scored_agent_pairs(round_, min_questions)
    status = "finalized" if board.finalized else "not finalized"
    out = [
        f"**{round_.round.name}** (`{round_.round.slug}`, project {round_.round.id}; "
        f"{status}, read {round_.fetched_at.date().isoformat()}; "
        f"{_fmt_questions(board.max_coverage)} questions scored at most).",
        "",
    ]
    if kept:
        out += [
            "| Model | Agent | Agent avg [interval] | Agent n | Template | "
            "Template avg [interval] | Template n | Ahead |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p in kept:
            out.append(
                f"| {p.model} | `{p.agent.name}` | {_fmt_avg(p.agent)} | "
                f"{_fmt_questions(p.agent.coverage)} | `{p.template.name}` | "
                f"{_fmt_avg(p.template.entry)} | {_fmt_questions(p.template.entry.coverage)} | "
                f"{p.ahead} |"
            )
        out.append("")
    ahead = sum(p.ahead == "agent" for p in kept)
    overlap = sum(p.intervals_overlap is True for p in kept)
    unknown = sum(p.intervals_overlap is None for p in kept)
    line = (
        f"Agent ahead in {ahead} of {len(kept)} pairs; intervals overlap in {overlap}"
        + (f" ({unknown} without an interval)" if unknown else "")
        + f". Pairs with both entries at {_fmt_questions(min_questions)} or more scored "
        "questions only"
    )
    if left:
        line += "; left out: " + ", ".join(
            f"{p.model} (agent {_fmt_questions(p.agent.coverage)}, template "
            f"{_fmt_questions(p.template.entry.coverage)})"
            for p in sorted(left, key=lambda p: p.model)
        )
    out.append(line + ".")
    return "\n".join(out) + "\n"


@dataclass(frozen=True)
class AdmittedRoute:
    template: TemplateRow
    listed_ids: tuple[str, ...]  # OpenRouter ids matched by name; empty: not listed
    priced_id: str | None
    usd_per_million_input: float | None
    usd_per_million_output: float | None
    per_question: float | None

    @property
    def per_round(self) -> float | None:
        return None if self.per_question is None else self.per_question * ROUND_QUESTIONS


def question_tokens(draws: int) -> tuple[int, int]:
    """Worst-case ``(input, output)`` tokens for one question: planner, brief, ``draws`` draws."""
    from augur.agents.forecaster import FORECAST_MAX_TOKENS
    from augur.agents.researcher import BRIEF_MAX_TOKENS, PLAN_MAX_TOKENS

    return (
        PLANNER_INPUT_TOKENS + BRIEF_INPUT_TOKENS + draws * DRAW_INPUT_TOKENS,
        PLAN_MAX_TOKENS + BRIEF_MAX_TOKENS + draws * FORECAST_MAX_TOKENS,
    )


def worst_case_cost(
    entries: Sequence[CatalogEntry], draws: int
) -> tuple[CatalogEntry, float] | None:
    """The dearest priced listing among ``entries`` and one question's worst-case cost on it."""
    tokens_in, tokens_out = question_tokens(draws)
    best: tuple[CatalogEntry, float] | None = None
    for e in entries:
        if e.usd_per_million_input is None or e.usd_per_million_output is None:
            continue
        cost = (tokens_in * e.usd_per_million_input + tokens_out * e.usd_per_million_output) / 1e6
        if best is None or cost > best[1]:
            best = (e, cost)
    return best


def admitted_by_template(
    round_: RoundStandings, catalog: Catalog, draws: int | None = None
) -> tuple[list[AdmittedRoute], list[TemplateRow], list[tuple[str, TemplateRow | None]]]:
    """The proposed floor: admit a model iff its plain template's average is >= 0.

    Returns ``(admitted, rejected, frontier)``. ``frontier`` lists every
    ``FRONTIER_TIER`` entry with its plain template (None: no template evidence).
    """
    from augur.llm.registry import FRONTIER_TIER, bare_model_name

    if draws is None:
        draws = int(Settings.model_fields["forecasters"].default)
    plain = _plain_by_key(_template_rows(round_))
    admitted: list[AdmittedRoute] = []
    rejected: list[TemplateRow] = []
    for row in sorted(plain.values(), key=lambda r: (-(r.average or 0.0), r.name)):
        if (row.average or 0.0) < 0:
            rejected.append(row)
            continue
        listed = catalog.same_model(row.key)
        priced = worst_case_cost(listed, draws)
        admitted.append(
            AdmittedRoute(
                template=row,
                listed_ids=tuple(sorted(e.id for e in listed)),
                priced_id=priced[0].id if priced else None,
                usd_per_million_input=priced[0].usd_per_million_input if priced else None,
                usd_per_million_output=priced[0].usd_per_million_output if priced else None,
                per_question=priced[1] if priced else None,
            )
        )
    keyed = {bare_model_name(k): r for k, r in plain.items()}
    frontier = [(name, keyed.get(name)) for name in sorted(FRONTIER_TIER)]
    return admitted, rejected, frontier


def add_parser(subparsers: Any) -> argparse.ArgumentParser:
    sp: argparse.ArgumentParser = subparsers.add_parser(
        "standings",
        help="Augur's rank in a MiniBench round beside rank 1, the template, the community",
        description=(
            "Per MiniBench round: Augur's rank and average spot peer score with interval, "
            "beside rank 1, the best template bot, the best Metaculus house bot, the best listed "
            "open-source bot and the community prediction, from Metaculus's leaderboard API; "
            "each question Augur missed is explained from its ledger."
        ),
    )
    sp.add_argument("--round", dest="round_slug", metavar="SLUG", help="e.g. minibench-2026-11-02")
    sp.add_argument(
        "--from", dest="from_path", type=Path, metavar="PATH", help="render a saved snapshot"
    )
    sp.add_argument(
        "--ledger", type=Path, action="append", default=[], help="explain misses from this ledger"
    )
    sp.add_argument("--out", type=Path, help="save the fetched snapshot as JSON here")
    return sp


async def main(args: argparse.Namespace, settings: Settings) -> int:
    ledgers: list[Path] = list(args.ledger or [])
    from_path: Path | None = args.from_path
    if from_path is not None:
        round_ = load_snapshot(from_path)
    else:
        if not args.round_slug:
            raise ConfigError("name a round with --round SLUG (e.g. minibench), or use --from PATH")
        limiter = RateLimiter(rate_per_second=1.0, burst=2, max_concurrent=2)
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http:
            client = MetaculusClient.from_settings(settings, http, limiter)
            username = await resolve_username(client, os.environ)
            round_ = await fetch_round(
                client, args.round_slug, SystemClock(), augur_username=username
            )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(snapshot_text(round_), encoding="utf-8")
    outcomes = read_outcomes(ledgers) if ledgers else None
    print(render(round_, outcomes), end="")
    if shows_agent_pairs(round_):
        print(f"\n### Agent harness against template, same model ({round_.round.slug})\n")
        print(agent_pairs_markdown(round_), end="")
    if round_.board().finalized:
        # Every template by model and the agent-minus-template gap for each model pair.
        from augur.llm.catalog import load_snapshot as load_catalog
        from augur.report import templates_markdown

        print(f"\n### Template bots and the agent harness ({round_.round.slug})\n")
        print(templates_markdown(round_, load_catalog()), end="")
    return 0


# -- the season across rounds (Q5) -----------------------------------------------------

# The template bot on the free model Augur's shadow round uses: the outside line Q1's
# unsubmitted comparison sits beside.
SAME_MODEL_TEMPLATE = "metac-nemotron-3-ultra+asknews"


def _named(board: Leaderboard, username: str) -> Entry | None:
    for e in board.entries:
        if e.user is not None and e.user.username == username:
            return e
    return None


def _cell(entry: Entry | None) -> str:
    if entry is None:
        return "none"
    rank = f"rank {entry.rank}" if entry.rank is not None else "unranked"
    return f"{_fmt_avg(entry)}, {rank}"


def _named_cell(entry: Entry | None) -> str:
    return "none" if entry is None else f"{entry.name} {_cell(entry)}"


def season_markdown(
    rounds: Sequence[RoundStandings], outcomes: Mapping[int, list[Outcome]] | None
) -> str:
    """One row per round snapshot: where the field stood, and where Augur did.

    Averages are spot peer points a question covered (Metaculus's sum divided by
    coverage); a round that is not finalized is a reading on the date fetched.
    """
    header = (
        "| Round | Status, as read | Entries | Questions scored | Rank 1 | Best template bot "
        f"| `{SAME_MODEL_TEMPLATE}` (Augur's free model) | Community (unweighted) "
        "| Augur | Questions Augur submitted |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines: list[str] = []
    ordered = sorted(rounds, key=lambda r: (r.round.start_date or "", r.round.slug))
    for r in ordered:
        board = r.board()
        top = rank_one(board)
        community = next(
            (e for e in community_aggregates(board) if e.aggregation_method == "unweighted"), None
        )
        mine = augur_entry(board, r.augur_username)
        submitted = (
            sum(explain(q, outcomes)[0] for q in r.questions) if outcomes is not None else None
        )
        status = "finalized" if board.finalized else "open"
        lines.append(
            f"| {r.round.name} (`{r.round.slug}`, project {r.round.id}) "
            f"| {status}, {r.fetched_at.date().isoformat()} | {len(board.entries)} "
            f"| {_fmt_questions(board.max_coverage)} "
            f"| {_named_cell(top)} "
            f"| {_named_cell(best_template(board))} | {_cell(_named(board, SAME_MODEL_TEMPLATE))} "
            f"| {_cell(community)} "
            f"| {'no entry' if mine is None else _cell(mine)} "
            f"| {'not read' if submitted is None else f'{submitted} of {len(r.questions)}'} |"
        )
    if not lines:
        return "No standings snapshot committed.\n"
    return header + "\n".join(lines) + "\n"
