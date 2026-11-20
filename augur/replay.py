"""The replay evaluation: the bot that ships, against the price, on resolved questions.

M5's arms (stated in PLAN.md before any run) on the same resolved binary
questions:

* **(a) price-only:** the market's own price as of the cutoff, no model;
* **(b) Augur without prices:** the forecaster draws, median;
* **(c) blend:** ``sigmoid(w * logit(b) + (1 - w) * logit(a))`` with
  ``w = 1/3`` (AIA Forecaster's weight, fixed in advance, never fitted);
* **(d) Augur with prices:** the same draws with the price shown as the
  brief's only source;

and the settings arms, re-aggregated from (b)'s own draws with no new calls:
one draw against three, and the median against the mean, the mean of
log-odds and extremized log-odds (factor 1.5).

Prices are recorded first (:func:`record_prices`) as dated market sources in
a ``backtest`` run whose tournament ends in ``:prices``; arm (d) reads them
back from that ledger (:func:`price_research`), so the model sees exactly the
recorded source and the leakage gate checks it. Scoring
(:func:`score_corpus`) is offline: it reads the ledgers and the committed
ForecastBench subset, and every interval is a seeded paired bootstrap.
"""

from __future__ import annotations

import math
import sqlite3
import statistics
import time
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from augur.agents.researcher import ResearchResult
from augur.clock import parse_iso
from augur.errors import SearchError
from augur.forecastbench import FBQuestion
from augur.leakage import RunMode, admit_source
from augur.ledger import Ledger
from augur.research.markets import STALE_AFTER, MarketsSearcher
from augur.research.quotes import MarketQuote
from augur.research.source import DateEvidence, Source
from augur.score import Estimate, ScoringError, brier, clustered_bootstrap, log_score, paired

BLEND_WEIGHT = 1.0 / 3.0
"""Augur's weight in the log-odds blend: AIA Forecaster's, fixed before any run."""
EXTREMIZE = 1.5
"""The extremizing factor of the settings arm, fixed before any run."""
CLIP = (0.01, 0.99)
LIVE_MARKETS = True
"""Whether ``augur forecast`` runs the markets searcher by default. M5's rule admits it:
on C2 the interval of Brier(d) - Brier(b) is [-0.144, -0.054], below 0 (RESULTS.md)."""
LIVE_BLEND = True
"""Whether a live binary forecast is blended with its own community prediction. M5's rule
admits it: on C2 the interval of Brier(c) - Brier(b) is [-0.157, -0.069], below 0."""
PRICED_PLATFORMS = ("manifold", "polymarket")
PRICES_SUFFIX = ":prices"
WITH_PRICES_SUFFIX = ":with-prices"

PRICE_BRIEF = (
    "No other research was done. The one source below is the price of the prediction "
    "market this question asks about, as it stood at the time of the question. Weigh it "
    "with what you know; you may disagree with it."
)


def clip(p: float) -> float:
    return min(max(p, CLIP[0]), CLIP[1])


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def blend(augur: float, price: float, weight: float = BLEND_WEIGHT) -> float:
    """The fixed log-odds blend of arm (c), both inputs clipped to [0.01, 0.99]."""
    return sigmoid(weight * logit(augur) + (1.0 - weight) * logit(price))


def mean_logodds(ps: Sequence[float], factor: float = 1.0) -> float:
    return sigmoid(factor * statistics.fmean(logit(p) for p in ps))


def exact_price(research: ResearchResult) -> MarketQuote | None:
    """The question's own community prediction among the research's sources, if any.

    Only an exact price of the same question is ever blended; a similar market
    found by search is shown to the model and never blended.
    """
    for source in research.sources:
        quote = source.quote
        if quote is not None and quote.match == "exact" and quote.platform == "metaculus":
            return quote
    return None


# -- recording prices ---------------------------------------------------------


@dataclass
class PriceReport:
    run_id: str
    priced: int = 0
    unpriced: dict[str, int] = field(default_factory=dict)

    def line(self) -> str:
        missing = ", ".join(f"{p} {n}" for p, n in sorted(self.unpriced.items())) or "none"
        return f"run {self.run_id}: {self.priced} priced; no usable price: {missing}"


async def quote_for(searcher: MarketsSearcher, fb: FBQuestion) -> tuple[MarketQuote | None, str]:
    """The exact as-of price of a ForecastBench market question, or why there is none."""
    if fb.source in PRICED_PLATFORMS:
        quote = await searcher.exact_quote(
            fb.source, fb.fb_id, fb.freeze_datetime, title=fb.question, url=fb.url
        )
        return (
            quote,
            "" if quote else "no bet before the freeze, or no history point in the week before it",
        )
    if fb.source == "metaculus":
        quote = await searcher.metaculus_quote(int(fb.fb_id), fb.freeze_datetime)
        return quote, "" if quote else "the aggregate history is hidden from the bot's token"
    return None, f"{fb.source} has no public price API"


async def record_prices(
    *,
    questions: Sequence[FBQuestion],
    tournament: str,
    searcher: MarketsSearcher,
    ledger: Ledger,
    code_version: str,
    out: Callable[[str], None] = print,
) -> PriceReport:
    """Fetch and record each question's exact price as of its freeze instant."""
    run_id = ledger.start_run(
        mode=RunMode.BACKTEST,
        tournament=tournament + PRICES_SUFFIX,
        provider="markets",
        model="none",
        settings={"questions": len(questions), "history_window_days": 7},
        code_version=code_version,
    )
    report = PriceReport(run_id)
    for i, fb in enumerate(questions, start=1):
        started = time.monotonic()
        q = fb.to_question()
        ledger.record_question(run_id, q, fb.freeze_datetime, {"fb_id": fb.fb_id})
        try:
            quote, why = await quote_for(searcher, fb)
        except SearchError as exc:
            quote, why = None, f"search failed: {exc}"[:300]
        if quote is None:
            report.unpriced[fb.source] = report.unpriced.get(fb.source, 0) + 1
            ledger.record_outcome(
                run_id=run_id,
                question_id=q.question_id,
                post_id=q.post_id,
                question_type="binary",
                status="skipped",
                reason=f"no usable price: {why}",
                submitted=False,
                closes_at=None,
                minutes_left=None,
                cost_usd=0.0,
                seconds=time.monotonic() - started,
            )
            out(f"[{i}/{len(questions)}] {fb.source}:{fb.fb_id} no price ({why})")
            continue
        source = searcher.to_source(quote, fb.question)
        ledger.record_source(source)
        ledger.record_admission(
            run_id, q.question_id, admit_source(source, fb.freeze_datetime, RunMode.BACKTEST)
        )
        report.priced += 1
        ledger.record_outcome(
            run_id=run_id,
            question_id=q.question_id,
            post_id=q.post_id,
            question_type="binary",
            status="forecast",
            reason=f"priced {quote.probability:.4f} at {quote.price_time.isoformat()}",
            submitted=False,
            closes_at=None,
            minutes_left=None,
            cost_usd=0.0,
            seconds=time.monotonic() - started,
        )
        freeze = "" if fb.freeze_value is None else f" (freeze value {fb.freeze_value:.4f})"
        out(f"[{i}/{len(questions)}] {fb.source}:{fb.fb_id} {quote.probability:.4f}{freeze}")
    return report


# -- reading them back -------------------------------------------------------------


def recorded_prices(path: Path, tournament: str) -> dict[int, Source]:
    """Each question's recorded price source from the ``:prices`` run of ``tournament``."""
    out: dict[int, Source] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT su.question_id, s.source_id, s.searcher, s.query, s.url, s.title, s.excerpt, "
            "s.content_sha256, s.published_at, s.retrieved_at, s.as_of, s.date_evidence, "
            "s.evidence_detail, m.platform, m.market_id, m.outcome, m.probability, "
            "m.price_time, m.match, m.volume, m.closes_at "
            "FROM source_uses su JOIN runs r ON r.run_id = su.run_id "
            "JOIN sources s ON s.source_id = su.source_id "
            "JOIN market_quotes m ON m.source_id = su.source_id "
            "WHERE r.tournament = ? AND su.admitted = 1 ORDER BY su.use_id",
            (tournament + PRICES_SUFFIX,),
        ).fetchall()
    for row in rows:
        as_of = parse_iso(row[10])
        quote = MarketQuote(
            platform=row[13],
            market_id=row[14],
            title=row[5],
            url=row[4],
            outcome=row[15],
            probability=row[16],
            price_time=parse_iso(row[17]),
            as_of=as_of,
            match=row[18],
            volume=row[19],
            closes_at=parse_iso(row[20]) if row[20] else None,
        )
        out[int(row[0])] = Source(
            source_id=row[1],
            searcher=row[2],
            query=row[3],
            url=row[4],
            title=row[5],
            excerpt=row[6],
            content_sha256=row[7],
            published_at=parse_iso(row[8]) if row[8] else None,
            retrieved_at=parse_iso(row[9]),
            as_of=as_of,
            date_evidence=DateEvidence(row[11]),
            evidence_detail=row[12],
            quote=quote,
        )
    return out


def price_research(
    prices: dict[int, Source],
) -> Callable[[FBQuestion], ResearchResult | None]:
    """Arm (d)'s research: the recorded price as the brief's only source."""

    def research(fb: FBQuestion) -> ResearchResult | None:
        source = prices.get(fb.question_id)
        if source is None:
            return None
        brief = f"{PRICE_BRIEF}\n\n[S1] {source.excerpt} ({source.url})"
        return ResearchResult(queries=[], sources=[source], brief=brief)

    return research


def _forecasts(path: Path, tournament: str) -> tuple[dict[int, float], dict[int, list[float]]]:
    """The latest aggregate and the members of that run, per question."""
    aggregates: dict[int, tuple[str, str, float]] = {}
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        for qid, run_id, made, p in conn.execute(
            "SELECT f.question_id, f.run_id, f.recorded_at, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.mode = 'backtest' AND r.tournament = ? AND f.kind = 'aggregate'",
            (tournament,),
        ):
            if int(qid) not in aggregates or made > aggregates[int(qid)][1]:
                aggregates[int(qid)] = (run_id, made, float(p))
        members: dict[int, list[float]] = {}
        for qid, run_id, p in conn.execute(
            "SELECT f.question_id, f.run_id, f.probability FROM forecasts f "
            "JOIN runs r ON r.run_id = f.run_id "
            "WHERE r.mode = 'backtest' AND r.tournament = ? AND f.kind = 'member' "
            "ORDER BY f.question_id, f.member",
            (tournament,),
        ):
            if int(qid) in aggregates and aggregates[int(qid)][0] == run_id:
                members.setdefault(int(qid), []).append(float(p))
    return {q: v[2] for q, v in aggregates.items()}, members


# -- scoring ---------------------------------------------------------------------


@dataclass(frozen=True)
class ArmScore:
    name: str
    brier: Estimate
    log: float


@dataclass(frozen=True)
class Diff:
    name: str
    brier: Estimate


@dataclass(frozen=True)
class CorpusScore:
    label: str
    questions: int
    priced: int
    priced_by_platform: dict[str, tuple[int, int]]
    """Platform: (priced, questions)."""
    freeze_match: tuple[int, int]
    """Priced questions whose as-of price is within 0.01 of ForecastBench's freeze value."""
    first_measure: tuple[int, int]
    """``freeze_match`` under the seven-day staleness cut the first price run applied to
    every platform, before Manifold was exempted after the data were seen: (matched,
    priced). Kept so P5 is reported as first measured as well as after the change."""
    arms: list[ArmScore]
    diffs: list[Diff]
    settings: list[Diff]
    settings_n: int
    all_b: ArmScore | None


def _score(name: str, ps: Sequence[float], ys: Sequence[int]) -> ArmScore:
    briers = [brier(clip(p), y) for p, y in zip(ps, ys, strict=True)]
    logs = [log_score(clip(p) if y else 1.0 - clip(p)) for p, y in zip(ps, ys, strict=True)]
    return ArmScore(name, clustered_bootstrap(briers), statistics.fmean(logs))


def _diff(name: str, p: Sequence[float], q: Sequence[float], ys: Sequence[int]) -> Diff:
    return Diff(
        name,
        paired(
            [brier(clip(x), y) for x, y in zip(p, ys, strict=True)],
            [brier(clip(x), y) for x, y in zip(q, ys, strict=True)],
        ),
    )


def score_corpus(
    *,
    label: str,
    questions: Sequence[FBQuestion],
    tournament: str,
    forecasts: Path,
    prices: Path,
    with_prices: Path | None = None,
) -> CorpusScore:
    """Every arm on one corpus, paired on the questions both sides of a row have."""
    quotes = {q: s.quote for q, s in recorded_prices(prices, tournament).items() if s.quote}
    b_all, members = _forecasts(forecasts, tournament)
    d_all: dict[int, float] = {}
    if with_prices is not None:
        d_all, _ = _forecasts(with_prices, tournament + WITH_PRICES_SUFFIX)
    outcome = {q.question_id: q.outcome for q in questions}
    by_platform: dict[str, tuple[int, int]] = {}
    match = 0
    first_match = first_priced = 0
    for fb in questions:
        got, total = by_platform.get(fb.source, (0, 0))
        priced = fb.question_id in quotes
        by_platform[fb.source] = (got + int(priced), total + 1)
        quote = quotes.get(fb.question_id)
        if (
            quote
            and fb.freeze_value is not None
            and abs(quote.probability - fb.freeze_value) <= 0.01
        ):
            match += 1
        if quote and quote.price_time >= quote.as_of - STALE_AFTER:
            first_priced += 1
            if fb.freeze_value is not None and abs(quote.probability - fb.freeze_value) <= 0.01:
                first_match += 1
    both = [q.question_id for q in questions if q.question_id in quotes and q.question_id in b_all]
    if not both:
        raise ScoringError(f"{label}: no question has both a price and a forecast")
    ys = [outcome[q] for q in both]
    a = [quotes[q].probability for q in both]
    b = [b_all[q] for q in both]
    c = [blend(x, y) for x, y in zip(b, a, strict=True)]
    arms = [_score("a", a, ys), _score("b", b, ys), _score("c", c, ys)]
    diffs = [_diff("b-a", b, a, ys), _diff("c-b", c, b, ys), _diff("c-a", c, a, ys)]
    with_d = [q for q in both if q in d_all]
    if with_d:
        yd = [outcome[q] for q in with_d]
        d = [d_all[q] for q in with_d]
        bd = [b_all[q] for q in with_d]
        ad = [quotes[q].probability for q in with_d]
        arms.append(_score("d", d, yd))
        diffs += [_diff("d-b", d, bd, yd), _diff("d-a", d, ad, yd)]
    full = [q for q in questions if q.question_id in b_all]
    all_b = _score("b", [b_all[q.question_id] for q in full], [q.outcome for q in full])
    three = [q for q in full if len(members.get(q.question_id, [])) >= 3]
    settings: list[Diff] = []
    if three:
        y3 = [q.outcome for q in three]
        ms = [members[q.question_id][:3] for q in three]
        median = [statistics.median(m) for m in ms]
        settings = [
            _diff("one draw - median of 3", [m[0] for m in ms], median, y3),
            _diff("mean - median", [statistics.fmean(m) for m in ms], median, y3),
            _diff("mean log-odds - median", [mean_logodds(m) for m in ms], median, y3),
            _diff(
                f"extremized x{EXTREMIZE} - median",
                [mean_logodds(m, EXTREMIZE) for m in ms],
                median,
                y3,
            ),
        ]
    return CorpusScore(
        label=label,
        questions=len(questions),
        priced=len(quotes),
        priced_by_platform=by_platform,
        freeze_match=(match, len(quotes)),
        first_measure=(first_match, first_priced),
        arms=arms,
        diffs=diffs,
        settings=settings,
        settings_n=len(three),
        all_b=all_b,
    )
