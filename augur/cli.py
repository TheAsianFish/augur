"""The ``augur`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import httpx

from augur import (
    baseline,
    diagram,
    fbcompare,
    floor,
    livecal,
    q4,
    q6,
    report,
    rereads,
    scorecard,
    shadow,
    standings,
    stateval,
)
from augur.backtest import (
    PLAN_SUFFIX,
    PinnedClock,
    Retrieval,
    plan_queries,
    read_queries,
    run_forecastbench,
    sample,
    write_queries,
)
from augur.clock import SystemClock, parse_iso
from augur.config import Settings
from augur.errors import AugurError, ConfigError, MissingCredentialsError
from augur.export import export_jsonl
from augur.forecastbench import FBQuestion, load_set
from augur.importer import import_exports
from augur.judge import judge_arm
from augur.leakage import scan_ledger
from augur.ledger import Ledger
from augur.llm.catalog import (
    SNAPSHOT_PATH,
    fetch_catalog,
    load_snapshot,
    price_for,
    release_bound,
    snapshot_text,
)
from augur.llm.endpoints import Role, Route, endpoints_from_env, parse_route
from augur.llm.factory import build_providers, build_router, default_http_client
from augur.llm.openai_compat import OpenAICompatProvider
from augur.llm.probe import (
    PROBE_TIMEOUT_S,
    measure_limit,
    new_session_id,
    openrouter_key_status,
    probe_routes,
    record_limit,
    record_probes,
)
from augur.llm.registry import FRONTIER_TIER_NAME, in_frontier_tier
from augur.log import configure_logging, get_logger
from augur.metaculus import MetaculusClient
from augur.ratelimit import RateLimiter
from augur.redraw import recorded_research, research_from
from augur.replay import (
    LIVE_BLEND,
    LIVE_MARKETS,
    WITH_PRICES_SUFFIX,
    price_research,
    record_prices,
    recorded_prices,
)
from augur.research.asknews import AskNewsSearcher
from augur.research.gdelt import GdeltSearcher, ResponseCache, gdelt_query
from augur.research.gkg import GkgIndex, build_index
from augur.research.markets import MarketsSearcher
from augur.research.series import SeriesFetcher
from augur.research.source import Searcher, Source
from augur.research.wayback import WaybackSearcher
from augur.research.wikipedia import WikipediaSearcher
from augur.resolve import (
    resolve_forecastbench,
    resolve_manual,
    resolve_manual_file,
    resolve_metaculus,
)
from augur.runner import TOURNAMENTS, ForecastRun, code_version, format_probe
from augur.statarm import load_base_rates
from augur.statlive import LiveStatArm

log = get_logger("augur.cli")
GDELT_CACHE = Path("data/backtest/gdelt-cache")
GKG_INDEX = Path("data/backtest/gdelt-gkg-index.tsv.gz")
GKG_MANIFEST = Path("data/backtest/gdelt-gkg-manifest.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augur", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser(
        "forecast",
        help="forecast open questions (a dry run unless --submit)",
        description=(
            "Forecast a tournament's open binary, multiple-choice, numeric and discrete "
            "questions, soonest-closing first. Without METACULUS_TOKEN it runs "
            "on the committed sample questions; with it, on the live listing. Nothing is "
            "posted unless --submit is given."
        ),
    )
    fc.add_argument(
        "--tournament",
        default="minibench",
        help=f"slug or id; aliases: {', '.join(TOURNAMENTS)} (default minibench)",
    )
    posting = fc.add_mutually_exclusive_group()
    posting.add_argument(
        "--submit", action="store_true", help="post forecasts and private comments"
    )
    posting.add_argument(
        "--shadow-ledger",
        type=Path,
        metavar="PATH",
        help="a shadow tick (M11): skip every question this ledger already holds an Augur "
        "forecast for, before any model call; never submits (excludes --submit)",
    )
    fc.add_argument("--samples", type=Path, help="forecast sample posts from this directory")
    fc.add_argument("--max-questions", type=int, help="override AUGUR_MAX_QUESTIONS_PER_RUN")
    fc.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    fc.add_argument("--no-asknews", action="store_true", help="skip AskNews even with keys")
    fc.add_argument(
        "--markets",
        action=argparse.BooleanOptionalAction,
        default=LIVE_MARKETS,
        help="the markets searcher: the question's own community prediction where the token "
        "sees it and similar Manifold, Polymarket and Kalshi markets, priced as of the cutoff "
        "(on by default since M5's replay evaluation; RESULTS.md)",
    )
    fc.add_argument(
        "--blend",
        action=argparse.BooleanOptionalAction,
        default=LIVE_BLEND,
        help="blend a binary forecast with the question's own community prediction, when the "
        "markets searcher finds it, one third to two thirds in log-odds (on by default since M5)",
    )
    fc.add_argument(
        "--calibrate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply the live calibrator (augur/live_calibrator.json) to binary forecasts from "
        "the model it was fitted on; other models' forecasts stay raw (M8), and so do "
        "questions from sources it is withheld from (Metaculus, so every live question)",
    )
    fc.add_argument(
        "--stat-arm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="write M12's statistical arm and its registered blend as the stat and blend "
        "reference rows of each question, before it closes (on by default in a shadow tick, "
        "off otherwise; nothing it writes is posted)",
    )
    fc.add_argument(
        "--no-probe", action="store_true", help="skip the opening model probe (AUGUR_PROBE=0)"
    )
    fc.add_argument(
        "--coverage",
        type=Path,
        help="write the run's coverage report (forecast, missed, skipped, failed) as JSON",
    )
    fc.add_argument(
        "--reforecast",
        action="store_true",
        help="forecast again questions this account already forecast (bot-testing-area only)",
    )

    models = sub.add_parser("models", help="the model chains: list, probe, measure limits")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser(
        "list", help="each route with its listed price, listing date and tier (no model calls)"
    )
    probe = models_sub.add_parser(
        "probe",
        help="call each listed model once and record which answer; --burst measures limits",
    )
    probe.add_argument(
        "--route",
        action="append",
        default=[],
        metavar="ENDPOINT:MODEL",
        help="probe these routes instead of the configured chains (repeatable)",
    )
    probe.add_argument(
        "--burst",
        type=int,
        default=0,
        metavar="N",
        help="then send up to N requests back to back to each free route that answered, "
        "stopping at the first refusal, to measure its real rate limit (at most 60)",
    )
    probe.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    snap = models_sub.add_parser(
        "snapshot", help="refresh the committed copy of OpenRouter's model list"
    )
    snap.add_argument("--out", type=Path, default=SNAPSHOT_PATH)

    bt = sub.add_parser("backtest", help="forecast resolved questions with the clock pinned")
    bt_sub = bt.add_subparsers(dest="backtest_command", required=True)
    fb = bt_sub.add_parser(
        "forecastbench",
        help="ForecastBench market questions, no retrieval, as of each freeze date",
        description=(
            "Forecast the eligible questions of one ForecastBench question set with the "
            "configured forecast model and no retrieval. Resumable: questions that already "
            "have an aggregate in the ledger are not forecast again."
        ),
    )
    fb.add_argument("--questions", type=Path, required=True, help="question set JSON")
    fb.add_argument("--resolutions", type=Path, required=True, help="resolution set JSON")
    fb.add_argument("--limit", type=int, help="forecast only the first N eligible questions")
    fb.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    fb.add_argument(
        "--retrieval",
        choices=("none", "open", "now"),
        default="none",
        help="none (the default); open: research each question as of its open instant "
        "(a backtest); now: research as of this moment, Q2's leaky arm (an unrestricted run, "
        "never scored as a backtest)",
    )
    fb.add_argument(
        "--searchers",
        default="wikipedia,wayback",
        help="comma-separated searchers for --retrieval (wikipedia, wayback, gdelt)",
    )
    fb.add_argument(
        "--add-to",
        type=Path,
        metavar="LEDGER",
        help="with --retrieval open: start from the sources arm O recorded in this ledger and "
        "add what --searchers find (M13's arm G); a question gaining nothing keeps O's forecast",
    )
    fb.add_argument(
        "--gdelt-cache",
        type=Path,
        default=GDELT_CACHE,
        help=f"where GDELT's answers are cached, one file per request (default {GDELT_CACHE})",
    )
    fb.add_argument(
        "--queries",
        type=Path,
        help="planned queries per question (JSON from --plan-out), shared by every arm",
    )
    fb.add_argument(
        "--plan-out",
        type=Path,
        help="only plan each question's queries as of its open instant and write them here",
    )
    fb.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="STEPS",
        help="with --retrieval: the bounded research loop with this many searches",
    )
    fb.add_argument(
        "--sample",
        type=int,
        help="only the first N eligible questions in question-id order (a fixed sample)",
    )
    fb.add_argument(
        "--prices",
        type=Path,
        help="arm (d): show each question's price recorded by `augur backtest prices` in this "
        "ledger as the brief's only source, and skip questions without one",
    )
    fb.add_argument(
        "--passes",
        type=int,
        choices=(1, 2),
        default=1,
        help="with --retrieval: 2 adds a second research pass on sub-questions",
    )
    fb.add_argument(
        "--research-from",
        type=Path,
        metavar="LEDGER",
        help="forecast from the briefs an arm recorded in this ledger (head count, model mix)",
    )
    fb.add_argument(
        "--research-arm",
        default="onepass-asof-open",
        help="with --research-from: the arm's label (default onepass-asof-open)",
    )
    fb.add_argument(
        "--first-member",
        type=int,
        default=0,
        help="number the draws from here (seed = member), to add to a recorded ensemble",
    )
    fb.add_argument(
        "--label",
        default="",
        help="with --research-from: a suffix naming this run's arm in the ledger",
    )
    gk = bt_sub.add_parser(
        "gkg-index",
        help="build M13's local GDELT index from the 15-minute GKG files (no model call)",
        description=(
            "Download every GDELT GKG 2.1 file with a batch stamp after --start and at or "
            "before --end, check each against the MD5 in GDELT's master list, and write one "
            "line per article (batch stamp, URL, page title) to --out as gzipped TSV, with a "
            "manifest. The gdelt searcher reads it when GDELT's DOC API refuses."
        ),
    )
    gk.add_argument("--start", required=True, help="exclusive, ISO 8601 UTC")
    gk.add_argument("--end", required=True, help="inclusive, ISO 8601 UTC (the cutoff)")
    gk.add_argument("--out", type=Path, default=GKG_INDEX, help=f"default {GKG_INDEX}")
    gk.add_argument("--manifest", type=Path, default=GKG_MANIFEST, help=f"default {GKG_MANIFEST}")
    bp = bt_sub.add_parser(
        "prices",
        help="record each ForecastBench market question's exact price as of its freeze date",
        description=(
            "Fetch each eligible question's price from its market's public API as of the "
            "freeze instant (Manifold's last bet, Polymarket's last price-history point, "
            "the Metaculus aggregate where the token sees it) and record it as a dated "
            "market source. No model is called. Free."
        ),
    )
    bp.add_argument("--questions", type=Path, required=True, help="question set JSON")
    bp.add_argument("--resolutions", type=Path, required=True, help="resolution set JSON")
    bp.add_argument("--ledger", type=Path, required=True, help="the ledger to write")

    jd = sub.add_parser(
        "judge",
        help="rate each source a backtest arm showed its forecasters for leakage (0 to 4)",
        description=(
            "Rate every source the forecasters of one arm were shown, with El Lahib et al.'s "
            "0 to 4 leakage rubric, using the first research model. The judge sees the "
            "resolution. Resumable: questions already judged are skipped."
        ),
    )
    jd.add_argument("--ledger", type=Path, required=True, help="the arm's ledger")
    jd.add_argument("--tournament", required=True, help="the arm's tournament label")
    jd.add_argument("--questions", type=Path, required=True, help="question set JSON")
    jd.add_argument("--resolutions", type=Path, required=True, help="resolution set JSON")
    res = sub.add_parser(
        "resolve",
        help="append resolutions (and references Metaculus exposes) to a ledger",
        description=(
            "Read each live or dry-run question in the ledger from Metaculus and append its "
            "resolution, the community aggregate and Metaculus's own scores where the API "
            "shows them; or resolve a backtest from a ForecastBench resolution set; or "
            "record one resolution read by hand; or a round's resolutions from a CSV file, "
            "all of the round's forecast questions or none."
        ),
    )
    res.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    how = res.add_mutually_exclusive_group()
    how.add_argument("--forecastbench", type=Path, help="a ForecastBench resolution set JSON")
    how.add_argument(
        "--manual", nargs=2, metavar=("POST_ID", "VALUE"), help="record one resolution by hand"
    )
    how.add_argument(
        "--manual-file",
        type=Path,
        metavar="F",
        help="a CSV with the header post_id,value,evidence (an evidence URL on every line); "
        "written only if no question of the --standings round that the ledger forecast is "
        "left without a resolution",
    )
    res.add_argument("--evidence", default="", help="where a --manual resolution was read")
    res.add_argument(
        "--standings",
        type=Path,
        metavar="S",
        help="the round's standings snapshot (augur standings --out); needed by --manual-file",
    )

    sc = sub.add_parser(
        "score",
        help="score resolved forecasts in one or more ledgers, with paired intervals",
    )
    sc.add_argument("ledgers", type=Path, nargs="+")
    sc.add_argument(
        "--cluster",
        choices=("question", "week", "source"),
        default="question",
        help="what the bootstrap resamples: questions, opening weeks or sources",
    )
    sc.add_argument(
        "--mode",
        action="append",
        choices=("live", "dry_run", "backtest", "unrestricted"),
        help="runs to score (repeatable); by default every mode but unrestricted",
    )
    sc.add_argument("--resamples", type=int, default=10_000)

    rep = sub.add_parser(
        "report", help="re-render RESULTS.md's generated blocks from the committed ledgers"
    )
    rep.add_argument("--path", type=Path, default=Path("RESULTS.md"))
    rep.add_argument(
        "--check", action="store_true", help="exit 1 if the file differs from a fresh render"
    )

    pw = sub.add_parser(
        "power",
        help="the M7 power simulation: questions needed to detect a planted improvement",
    )
    pw.add_argument("--out", type=Path, default=report.POWER_PATH)
    pw.add_argument(
        "--check", action="store_true", help="exit 1 if the committed file differs from a re-run"
    )

    rr = sub.add_parser(
        "rereads",
        help="M14, synthetic: how often re-reading the one-round floor rule admits a "
        "candidate with no lead",
    )
    rr.add_argument("--out", type=Path, default=report.REREADS_PATH)
    rr.add_argument(
        "--check", action="store_true", help="exit 1 if the committed file differs from a re-run"
    )

    cal = sub.add_parser(
        "calibrate",
        help="Q4: raw against fixed extremizing, Platt and isotonic, fitted only on history "
        "that resolved before each question opened",
        description=(
            "Score M8's calibrators on corpus C4 (walk-forward and learning curve) and write "
            "the snapshot RESULTS.md renders from, with the reliability diagram beside it. "
            "Offline, free, deterministic."
        ),
    )
    cal.add_argument("--out", type=Path, default=q4.SNAPSHOT)
    cal.add_argument(
        "--check", action="store_true", help="exit 1 if the committed files differ from a re-run"
    )

    sa = sub.add_parser(
        "statarm",
        help="M12: the statistical arm and its blends on C4, with no model call",
        description=(
            "Score the model-free arm (price where readable, else the source's base rate) "
            "and the registered blends against Augur's raw forecast in M8's walk-forward "
            "harness on corpus C4, by source, and write the snapshot RESULTS.md renders "
            "from and the base rates the live arm uses. Offline, free, deterministic."
        ),
    )
    sa.add_argument("--out", type=Path, default=stateval.SNAPSHOT)
    sa.add_argument(
        "--coverage",
        choices=sorted(stateval.COVERAGE),
        metavar="ROUND",
        help="instead, run the series rule on a round's question list (posts read from the "
        "API with METACULUS_TOKEN, closes from the series source as of each question's "
        f"open) and write its coverage file; one of {', '.join(sorted(stateval.COVERAGE))}",
    )
    sa.add_argument(
        "--check", action="store_true", help="exit 1 if the committed files differ from a re-run"
    )

    dis = sub.add_parser(
        "q6",
        help="Q6: does disagreement among the fleet's forecasts predict its errors?",
        description=(
            "M9: AUROC of each disagreement measure for large errors, and shrink and skip "
            "policies against constant shrinkage and the source-aware calibrator, on draws "
            "already in the ledgers; then the pre-registered live half's counts. Offline, "
            "free, deterministic."
        ),
    )
    dis.add_argument("--out", type=Path, default=q6.SNAPSHOT)
    dis.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed snapshot differs from a re-run",
    )

    fbc = sub.add_parser(
        "fbcompare",
        help="M10: Augur's backtests beside every ForecastBench entrant on the same questions",
        description=(
            "Pair each ForecastBench entrant's published forecasts (the committed extract in "
            "data/forecastbench/forecast-sets) with Augur's arms C1, C2, O and C4 on the "
            "instances Augur scored, and read P30 to P33. Offline, free, deterministic."
        ),
    )
    fbc.add_argument("--out", type=Path, default=fbcompare.SNAPSHOT)
    fbc.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed snapshot differs from a re-run",
    )

    standings.add_parser(sub)
    baseline.add_parser(sub)
    floor.add_parser(sub)

    led = sub.add_parser("ledger", help="inspect a ledger")
    led_sub = led.add_subparsers(dest="ledger_command", required=True)
    check = led_sub.add_parser("check", help="scan a ledger for leakage; exit 1 on a violation")
    check.add_argument("paths", type=Path, nargs="+")
    export = led_sub.add_parser(
        "export", help="write a ledger as JSONL, one row per line, byte-stable"
    )
    export.add_argument("db", type=Path)
    export.add_argument("out", type=Path)
    imp = led_sub.add_parser(
        "import",
        help="rebuild a ledger from JSONL exports (files or directories, e.g. the ledger branch)",
    )
    imp.add_argument("paths", type=Path, nargs="+")
    imp.add_argument("--out", type=Path, required=True, help="the SQLite ledger to write")
    return parser


def _shadow_held(shadow_ledger: Path | None, ledger_path: Path) -> frozenset[int] | None:
    """A shadow tick's held question ids, or ``None`` for an ordinary run."""
    if shadow_ledger is None:
        return None
    if shadow_ledger.resolve() == ledger_path.resolve():
        raise AugurError(
            "--shadow-ledger must not be the run's own ledger; write the run to a fresh "
            "file and pass the ledger rebuilt from the shadow exports"
        )
    held = shadow.held_questions(shadow_ledger)
    print(f"shadow tick: {len(held)} question(s) already held in {shadow_ledger}")
    return held


async def _forecast(args: argparse.Namespace, settings: Settings) -> int:
    clock = SystemClock()
    ledger_path: Path = args.ledger or settings.ledger_path
    max_questions = (
        args.max_questions if args.max_questions is not None else settings.max_questions_per_run
    )
    if args.submit and not settings.has_metaculus_token:
        raise MissingCredentialsError("--submit", ("METACULUS_TOKEN",))
    held = _shadow_held(args.shadow_ledger, ledger_path)
    stat_on = args.stat_arm if args.stat_arm is not None else held is not None
    if stat_on and args.submit:
        raise AugurError(
            "--stat-arm runs only on unposted runs (shadow ticks and dry runs): nothing it "
            "writes may delay a submission; drop --stat-arm or --submit"
        )
    async with default_http_client() as http:
        # A replay must not depend on the network, prices included.
        catalog = load_snapshot() if settings.is_replay else await fetch_catalog(http, clock.now())
        router = build_router(settings, http, catalog, clock)
        probe_providers = None
        if settings.probe and not args.no_probe and not settings.is_replay:
            probe_providers = build_providers(
                settings, http, max_retries=0, record=False, timeout=PROBE_TIMEOUT_S
            )
        searchers: list[Searcher] = [WikipediaSearcher(http, clock)]
        if settings.has_asknews and not args.no_asknews:
            searchers.append(AskNewsSearcher.from_settings(settings, http, clock))
        else:
            log.info("asknews skipped", reason="no ASKNEWS_CLIENT_ID/ASKNEWS_CLIENT_SECRET")
        client = (
            MetaculusClient.from_settings(settings, http) if settings.has_metaculus_token else None
        )
        if args.markets and not settings.is_replay:
            searchers.append(MarketsSearcher(http, clock, metaculus=client))
        if client is None:
            print("METACULUS_TOKEN is not set: dry run on the committed sample questions.")
        stat_arm = (
            LiveStatArm(
                rates=load_base_rates(),
                fetcher=None if settings.is_replay else SeriesFetcher(http, clock),
            )
            if stat_on
            else None
        )
        with Ledger.open(ledger_path, clock) as ledger:
            run = ForecastRun(
                settings=settings,
                clock=clock,
                ledger=ledger,
                router=router,
                searchers=searchers,
                client=client,
                out=sys.stdout,
                probe_providers=probe_providers,
                blend_exact_price=bool(args.markets and args.blend),
                calibrator=livecal.load_live() if args.calibrate else None,
                stat_arm=stat_arm,
            )
            try:
                report = await run.run(
                    args.tournament,
                    submit=args.submit,
                    samples=args.samples,
                    max_questions=max_questions,
                    reforecast=args.reforecast,
                    shadow=held,
                )
            finally:
                await router.aclose()
        print(f"ledger: {ledger_path} (run {report.run_id})")
        if args.coverage is not None:
            args.coverage.parent.mkdir(parents=True, exist_ok=True)
            args.coverage.write_text(
                json.dumps(report.coverage(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
            )
        return 1 if report.count("failed") else 0


def _routes(
    args: argparse.Namespace, settings: Settings
) -> tuple[list[Route], dict[Route, list[str]]]:
    routes: list[Route] = []
    roles: dict[Route, list[str]] = {}
    if getattr(args, "route", None):
        endpoints = settings.endpoints or endpoints_from_env({})
        for text in args.route:
            route = parse_route(text, endpoints)
            routes.append(route)
            roles[route] = ["probe"]
        return routes, roles
    chains = settings.chains()
    order: tuple[Role, ...] = ("forecast", "research")
    for role in order:
        for route in chains[role]:
            if route not in routes:
                routes.append(route)
            roles.setdefault(route, []).append(role)
    return routes, roles


async def _models(args: argparse.Namespace, settings: Settings) -> int:
    clock = SystemClock()
    if settings.is_replay:
        raise ConfigError("AUGUR_PROVIDER=replay has no live models to list or probe")
    async with default_http_client() as http:
        catalog = await fetch_catalog(http, clock.now())
        if args.models_command == "snapshot":
            if catalog.origin != "live":
                raise AugurError("OpenRouter's model list could not be fetched; nothing written")
            args.out.write_text(snapshot_text(catalog), encoding="utf-8")
            print(f"wrote {len(catalog.entries)} models to {args.out}")
            return 0
        routes, roles = _routes(args, settings)
        print(
            f"model list: {catalog.origin}, {len(catalog.entries)} models, "
            f"fetched {catalog.fetched_at:%Y-%m-%d %H:%M} UTC; floor tier {FRONTIER_TIER_NAME}"
        )
        if args.models_command == "list":
            for route in routes:
                price = price_for(route, catalog)
                listed, basis = release_bound(route, catalog)
                print(
                    f"  {route.key:<48} "
                    + f"effort {route.reasoning_effort or '-':<7} "
                    + (
                        f"${price.usd_per_million_input:g}/${price.usd_per_million_output:g} per M"
                        if price
                        else "unpriced (not called)"
                    )
                    + f"  listed {listed or '?'} ({basis})"
                    + f"  {'frontier' if in_frontier_tier(route) else 'below tier'}"
                    + f"  [{'+'.join(roles[route])}]"
                )
            return 0
        providers = build_providers(
            settings.model_copy(
                update={
                    "forecast_models": ",".join(r.key for r in routes),
                    "research_models": None,
                }
            ),
            http,
            max_retries=0,
            record=False,
            timeout=PROBE_TIMEOUT_S,
        )
        for endpoint in {r.endpoint.name: r.endpoint for r in routes}.values():
            key = settings.keys.get(endpoint.key_env) if endpoint.key_env else None
            status = await openrouter_key_status(http, endpoint, key)
            if status is not None:
                print(f"{endpoint.name} key: {json.dumps(status, sort_keys=True)}")
        results = await probe_routes(routes, roles, providers, catalog)
        ledger_path: Path = args.ledger or settings.ledger_path
        session = new_session_id()
        with Ledger.open(ledger_path, clock) as ledger:
            record_probes(ledger, results, session, None)
            print(format_probe(results))
            cap = min(max(args.burst, 0), 60)
            for r in results:
                if not cap or not r.answers or r.route.endpoint.is_local:
                    continue
                if (
                    r.price is None
                    or r.price.usd_per_million_input
                    or r.price.usd_per_million_output
                ):
                    print(f"  burst skipped for {r.route.key}: not free")
                    continue
                key = (
                    settings.keys.get(r.route.endpoint.key_env)
                    if r.route.endpoint.key_env
                    else None
                )
                unthrottled = OpenAICompatProvider(
                    http,
                    r.route.endpoint,
                    key,
                    limiter=RateLimiter(1000, burst=1000, max_concurrent=1),
                    max_retries=0,
                )
                started = clock.now()
                limit = await measure_limit(r.route, unthrottled, catalog, cap)
                record_limit(ledger, limit, session, started)
                rpm = limit.requests_per_minute
                refusal = (
                    f"refused at request {limit.first_refusal_at} "
                    f"(HTTP {limit.refusal_status}, {limit.refusal_kind}, "
                    f"retry after {limit.refusal_retry_after}s)"
                    if limit.first_refusal_at
                    else f"no refusal in {limit.sent}"
                )
                print(
                    f"  burst {r.route.key}: {limit.ok}/{limit.sent} accepted in "
                    f"{limit.seconds:.1f} s ({rpm:.0f}/min); {refusal}"
                    if rpm is not None
                    else f"  burst {r.route.key}: {refusal}"
                )
                if limit.limit_headers:
                    print(f"    headers: {limit.limit_headers}")
        print(f"ledger: {ledger_path} (session {session})")
        answering = [r for r in results if r.answers and "forecast" in r.roles]
        return 0 if answering or args.route else 1


async def _backtest_prices(args: argparse.Namespace, settings: Settings) -> int:
    report = load_set(args.questions, args.resolutions, released=date.min)
    print("\n".join(report.lines()))
    clock = SystemClock()
    async with default_http_client() as http:
        metaculus = MetaculusClient.from_settings(settings, http)
        searcher = MarketsSearcher(http, clock, metaculus=metaculus)
        with Ledger.open(args.ledger, clock) as ledger:
            result = await record_prices(
                questions=report.kept,
                tournament=f"forecastbench:{report.question_set}",
                searcher=searcher,
                ledger=ledger,
                code_version=code_version(),
            )
    print(f"ledger: {args.ledger} {result.line()}")
    return 0


async def _backtest(args: argparse.Namespace, settings: Settings) -> int:
    if args.backtest_command == "prices":
        return await _backtest_prices(args, settings)
    if args.backtest_command == "gkg-index":
        start, end = parse_iso(args.start), parse_iso(args.end)
        async with default_http_client() as http:
            built = await build_index(http, start, end, args.out)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(built.manifest(start, end), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{built.files} GKG files, {len(built.missing)} missing from the master list, "
            f"{built.articles} articles with a title; wrote {args.out} and {args.manifest}"
        )
        return 0
    researching = args.plan_out is not None or args.retrieval != "none"
    if researching and (args.research_from is not None or args.first_member):
        raise ConfigError(
            "--research-from and --first-member reuse recorded research; "
            "do not combine them with --retrieval or --plan-out"
        )
    if args.passes != 1 and args.retrieval == "none":
        raise ConfigError(
            "--passes 2 needs --retrieval open or now: there is no research to deepen"
        )
    if settings.is_replay:
        raise ConfigError("a backtest needs a live model; set AUGUR_PROVIDER or the chains")
    answering = settings.chains()["forecast"][0]
    released, _ = release_bound(answering, load_snapshot())
    if released is None:
        raise ConfigError(
            f"{answering.key} has no known release date, so a backtest cannot rule out that "
            "it remembers the outcomes; use a model with one (augur models list)"
        )
    report = load_set(args.questions, args.resolutions, released=released, limit=args.limit)
    print("\n".join(report.lines()))
    questions = sample(report.kept, args.sample)
    if args.sample is not None:
        print(f"  sample  {len(questions):>4} (first {args.sample} by question id)")
    clock = PinnedClock(SystemClock().now())
    ledger_path: Path = args.ledger or settings.ledger_path
    tournament = f"forecastbench:{report.question_set}"
    research = None
    extra: dict[str, object] = {}
    if args.plan_out is not None or args.retrieval != "none":
        if args.prices is not None:
            raise ConfigError("--prices is its own arm; do not combine it with --retrieval")
        return await _backtest_research(args, settings, questions, tournament, ledger_path)
    if args.research_from is not None:
        if args.prices is not None:
            raise ConfigError("--research-from and --prices are separate arms")
        arm = f"{tournament}+{args.research_arm}"
        recorded = recorded_research(args.research_from, arm)
        print(f"{len(recorded)} recorded brief(s) of {arm} in {args.research_from}")
        research = research_from(recorded)
        extra = {"retrieval": f"recorded: {arm}", "research_ledger": str(args.research_from)}
        tournament = arm + (f"+{args.label}" if args.label else "+redraw")
    elif args.prices is not None:
        prices = recorded_prices(args.prices, tournament)
        print(f"{len(prices)} recorded prices in {args.prices}")
        research = price_research(prices)
        extra = {"retrieval": "recorded market price", "prices_ledger": str(args.prices)}
        tournament += WITH_PRICES_SUFFIX
    async with default_http_client() as http:
        # Rate-limit cool-downs run on the wall clock: a pinned clock never
        # reaches the end of a cool-down, so one 429 would fail every later question.
        router = build_router(settings, http, load_snapshot(), SystemClock())
        with Ledger.open(ledger_path, clock) as ledger:
            try:
                result = await run_forecastbench(
                    questions=questions,
                    tournament=tournament,
                    settings=settings,
                    router=router,
                    ledger=ledger,
                    clock=clock,
                    code_version=code_version(),
                    research_for=research,
                    extra_settings=extra,
                    first_member=args.first_member,
                )
            finally:
                await router.aclose()
    print(
        f"ledger: {ledger_path} (run {result.run_id}): {result.forecast} forecast, "
        f"{result.resumed} done earlier, {result.skipped} skipped, {result.failed} failed"
    )
    return 1 if result.failed else 0


def _gkg_index(path: Path, queries: dict[int, list[str]] | None) -> GkgIndex | None:
    """The local GKG index, if built, holding only words the planned queries can ask for."""
    if not path.exists() or not queries:
        return None
    vocabulary = {
        w.lower()
        for qs in queries.values()
        for q in qs
        for w in (gdelt_query(q) or "").split()
        if ":" not in w
    }
    index = GkgIndex.load(path, vocabulary)
    print(f"GKG index {path}: loaded for {len(vocabulary)} query words")
    return index


def _backtest_searchers(
    names: str,
    http: httpx.AsyncClient,
    gdelt_cache: Path = GDELT_CACHE,
    gkg: GkgIndex | None = None,
) -> list[Searcher]:
    # Searchers stamp ``retrieved_at`` with the real time; ``as_of`` is the cutoff.
    clock = SystemClock()
    found: list[Searcher] = []
    for name in (n.strip() for n in names.split(",") if n.strip()):
        if name == "wikipedia":
            found.append(WikipediaSearcher(http, clock))
        elif name == "wayback":
            found.append(WaybackSearcher(http, clock))
        elif name == "gdelt":
            found.append(
                GdeltSearcher(
                    http,
                    clock,
                    WaybackSearcher(http, clock),
                    cache=ResponseCache(gdelt_cache),
                    index=gkg,
                )
            )
        else:
            raise ConfigError(
                f"unknown backtest searcher {name!r}; use wikipedia, wayback and/or gdelt "
                "(AskNews and markets cannot prove dates for a backtest here)"
            )
    return found


async def _backtest_research(
    args: argparse.Namespace,
    settings: Settings,
    questions: list[FBQuestion],
    tournament: str,
    ledger_path: Path,
) -> int:
    """Q2's arms and the research loop: plan, or research as of open or now, then forecast."""
    queries: dict[int, list[str]] | None = None
    if args.queries is not None:
        queries = read_queries(args.queries)
        missing = [q for q in questions if q.question_id not in queries]
        print(f"{len(queries)} planned question(s) in {args.queries}; {len(missing)} unplanned")
    base: dict[int, list[Source]] | None = None
    if args.add_to is not None:
        if args.retrieval != "open" or args.loop or args.passes != 1 or queries is None:
            raise ConfigError(
                "--add-to is M13's arm G: one pass as of open with --queries, nothing else"
            )
        arm = f"{tournament}+onepass-asof-open"
        recorded = recorded_research(args.add_to, arm)
        base = {qid: list(r.research.sources) for qid, r in recorded.items()}
        missing_base = [q.question_id for q in questions if q.question_id not in base]
        if missing_base:
            raise ConfigError(
                f"{len(missing_base)} question(s) have no recorded research of {arm} in "
                f"{args.add_to}; arm G adds to O's evidence and cannot start without it"
            )
        print(f"{len(base)} recorded research(es) of {arm} in {args.add_to}")
    clock = PinnedClock(SystemClock().now())
    async with default_http_client() as http:
        router = build_router(settings, http, load_snapshot(), SystemClock())
        try:
            with Ledger.open(ledger_path, clock) as ledger:
                if args.plan_out is not None:
                    planned = await plan_queries(
                        questions=questions,
                        tournament=tournament + PLAN_SUFFIX,
                        settings=settings,
                        router=router,
                        ledger=ledger,
                        clock=clock,
                        code_version=code_version(),
                    )
                    write_queries(args.plan_out, planned)
                    print(f"planned {len(planned)} question(s); wrote {args.plan_out}")
                    return 0
                retrieval = Retrieval(
                    searchers=_backtest_searchers(
                        args.searchers,
                        http,
                        args.gdelt_cache,
                        _gkg_index(GKG_INDEX, queries) if "gdelt" in args.searchers else None,
                    ),
                    as_of="now" if args.retrieval == "now" else "open",
                    queries=queries,
                    loop_steps=args.loop,
                    passes=args.passes,
                    base=base,
                    per_query=1 if base is not None else 2,
                )
                result = await run_forecastbench(
                    questions=questions,
                    tournament=tournament + "+" + retrieval.label(),
                    settings=settings,
                    router=router,
                    ledger=ledger,
                    clock=clock,
                    code_version=code_version(),
                    retrieval=retrieval,
                    wall_clock=SystemClock(),
                    extra_settings={"queries_file": str(args.queries) if args.queries else None},
                )
        finally:
            await router.aclose()
    print(
        f"ledger: {ledger_path} (run {result.run_id}): {result.forecast} forecast, "
        f"{result.resumed} done earlier, {result.skipped} skipped, {result.failed} failed"
    )
    return 1 if result.failed else 0


async def _judge(args: argparse.Namespace, settings: Settings) -> int:
    report = load_set(args.questions, args.resolutions, released=date.min)
    clock = SystemClock()
    async with default_http_client() as http:
        router = build_router(settings, http, load_snapshot(), clock)
        try:
            with Ledger.open(args.ledger, clock) as ledger:
                result = await judge_arm(
                    ledger=ledger,
                    tournament=args.tournament,
                    questions=report.kept,
                    settings=settings,
                    router=router,
                    clock=clock,
                    code_version=code_version(),
                )
        finally:
            await router.aclose()
    print(
        f"judge run {result.run_id}: {result.judged} judged, {result.resumed} done earlier, "
        f"{len(result.failed)} failed of {result.questions}"
    )
    for line in result.failed:
        print(f"  failed: {line}")
    return 1 if result.failed else 0


async def _resolve(args: argparse.Namespace, settings: Settings) -> int:
    clock = SystemClock()
    ledger_path: Path = args.ledger or settings.ledger_path
    if not ledger_path.exists():  # noqa: ASYNC240 - one stat before any I/O starts
        raise AugurError(f"no ledger at {ledger_path}")
    with Ledger.open(ledger_path, clock) as ledger:
        if args.manual:
            post, value = args.manual
            status, recorded = resolve_manual(
                ledger, post_id=int(post), value=value, evidence=args.evidence
            )
            print(f"post {post}: {status} {recorded or ''}".rstrip())
            return 0
        if args.manual_file is not None:
            snapshot = standings.load_snapshot(args.standings)
            filed = resolve_manual_file(
                ledger, args.manual_file, [q.post_id for q in snapshot.questions]
            )
            for m in filed.written:
                print(f"post {m.post_id}: {m.status} {m.value or ''}".rstrip())
            print(filed.line(args.manual_file))
            return 0
        if args.forecastbench is not None:
            report = resolve_forecastbench(ledger, args.forecastbench)
            print(report.line())
            return 0
        if not settings.has_metaculus_token:
            raise MissingCredentialsError("augur resolve", ("METACULUS_TOKEN",))
        async with default_http_client() as http:
            client = MetaculusClient.from_settings(settings, http)
            report = await resolve_metaculus(ledger, client)
        print(report.line())
        return 1 if report.failed else 0


def _score(args: argparse.Namespace) -> int:
    for path in args.ledgers:
        if not path.exists():
            raise AugurError(f"no ledger at {path}")
    loaded = scorecard.load(args.ledgers, modes=args.mode)
    for problem in loaded.unscorable:
        print(f"not scored: {problem}")
    if not loaded.scored:
        print(
            f"nothing to score: {loaded.unresolved} forecast questions have no resolution and "
            f"{loaded.annulled} were annulled (run augur resolve first)"
        )
        return 1
    summary = scorecard.summarise(loaded.scored, cluster=args.cluster, resamples=args.resamples)
    print(scorecard.format_summary(summary, loaded))
    return 0


def _rereads(args: argparse.Namespace) -> int:
    text = rereads.snapshot_text(rereads.snapshot())
    out: Path = args.out
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} differs from a fresh simulation; run augur rereads")
            return 1
        print(f"{out} matches a fresh simulation")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _power(args: argparse.Namespace) -> int:
    snap = report.power_snapshot(Path.cwd())
    if snap is None:
        raise AugurError("the M7 arms' ledgers are not all in data/ledgers; run them first")
    text = json.dumps(snap, indent=1) + "\n"
    out: Path = args.out
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} differs from a fresh simulation; run augur power")
            return 1
        print(f"{out} matches a fresh simulation")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _q6(args: argparse.Namespace) -> int:
    snap = q6.snapshot(Path.cwd(), SystemClock().now().date())
    text = json.dumps(snap, indent=1, sort_keys=True) + "\n"
    out: Path = args.out
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} differs from a fresh run; run augur q6")
            return 1
        print(f"{out} matches a fresh run")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def _fbcompare(args: argparse.Namespace) -> int:
    text = fbcompare.dumps(fbcompare.snapshot(Path.cwd()))
    out: Path = args.out
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} differs from a fresh run; run augur fbcompare")
            return 1
        print(f"{out} matches a fresh run")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


async def _statarm_coverage(label: str, settings: Settings) -> int:
    if not settings.has_metaculus_token:
        raise MissingCredentialsError("augur statarm --coverage", ("METACULUS_TOKEN",))
    snap_path, out = stateval.COVERAGE[label]
    snap = standings.load_snapshot(snap_path)
    async with default_http_client() as http:
        client = MetaculusClient.from_settings(settings, http)
        data = await stateval.round_coverage(snap, client, SeriesFetcher(http, SystemClock()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stateval.dumps(data), encoding="utf-8")
    print(
        f"{label}: {data['questions']} questions, {data['numeric_or_discrete']} numeric or "
        f"discrete, the series rule reaches {data['reached']} and forecasts {data['forecast']}"
    )
    for row in data["rows"]:
        if "query" in row:
            print(f"  {row['post_id']} {row['query']}: {row['outcome']}")
    print(f"wrote {out}")
    return 0


def _statarm(args: argparse.Namespace, settings: Settings) -> int:
    if args.coverage:
        return asyncio.run(_statarm_coverage(args.coverage, settings))
    written = stateval.files(Path.cwd(), args.out)
    if args.check:
        stale = [p for p, text in written.items() if not p.exists() or p.read_text("utf-8") != text]
        for p in stale:
            print(f"{p} differs from a fresh run; run augur statarm")
        for snap_path, cov in stateval.COVERAGE.values():
            if cov.exists():
                problems = stateval.check_coverage(cov, standings.load_snapshot(snap_path))
                stale += [cov] if problems else []
                for line in problems:
                    print(line)
        if not stale:
            print(f"{args.out} and the live base rates match a fresh run")
        return 1 if stale else 0
    for p, text in written.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    out: Path = args.out
    snap = q4.snapshot(Path.cwd())
    files = {
        out: json.dumps(snap, indent=1, sort_keys=True) + "\n",
        out.with_suffix(".svg"): diagram.reliability_svg(snap),
        livecal.LIVE_PATH: json.dumps(
            q4.live_calibrator(Path.cwd(), snap), indent=1, sort_keys=True
        )
        + "\n",
    }
    if args.check:
        stale = [p for p, text in files.items() if not p.exists() or p.read_text("utf-8") != text]
        for p in stale:
            print(f"{p} differs from a fresh run; run augur calibrate")
        if not stale:
            print(f"{out}, its diagram and the live calibrator match a fresh run")
        return 1 if stale else 0
    out.parent.mkdir(parents=True, exist_ok=True)
    for p, text in files.items():
        p.write_text(text, encoding="utf-8")
        print(f"wrote {p}")
    return 0


def _report(args: argparse.Namespace) -> int:
    path: Path = args.path
    if not path.exists():
        raise AugurError(f"no {path}; run from the repository root or pass --path")
    root = path.resolve().parent
    if args.check:
        stale = report.check(path, root)
        if stale:
            print(f"{path} is stale in: {', '.join(stale)}; run augur report")
            return 1
        print(f"{path} matches a fresh render")
        return 0
    text = path.read_text(encoding="utf-8")
    fresh = report.render(text, root)
    if fresh != text:
        path.write_text(fresh, encoding="utf-8")
    print(f"{path}: {'re-rendered' if fresh != text else 'already current'}")
    return 0


def _ledger_check(paths: Sequence[Path]) -> int:
    bad = 0
    for path in paths:
        if not path.exists():
            raise AugurError(f"no ledger at {path}")
        violations = scan_ledger(path)
        print(f"{path}: {len(violations)} violation(s)")
        for v in violations:
            print(f"  {v.kind} run={v.run_id} question={v.question_id}: {v.detail}")
        bad += len(violations)
    return 1 if bad else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "resolve" and (args.manual_file is None) != (args.standings is None):
        parser.error("augur resolve: --manual-file and --standings go together")
    try:
        settings = Settings.from_env(os.environ)
        configure_logging(settings.log_level)
        if args.command == "forecast":
            return asyncio.run(_forecast(args, settings))
        if args.command == "standings":
            return asyncio.run(standings.main(args, settings))
        if args.command == "models":
            return asyncio.run(_models(args, settings))
        if args.command == "backtest":
            return asyncio.run(_backtest(args, settings))
        if args.command == "judge":
            return asyncio.run(_judge(args, settings))
        if args.command == "baseline":
            return asyncio.run(baseline.main(args, settings))
        if args.command == "floor":
            return floor.main(args)
        if args.command == "resolve":
            return asyncio.run(_resolve(args, settings))
        if args.command == "score":
            return _score(args)
        if args.command == "report":
            return _report(args)
        if args.command == "power":
            return _power(args)
        if args.command == "rereads":
            return _rereads(args)
        if args.command == "calibrate":
            return _calibrate(args)
        if args.command == "statarm":
            return _statarm(args, settings)
        if args.command == "q6":
            return _q6(args)
        if args.command == "fbcompare":
            return _fbcompare(args)
        if args.ledger_command == "import":
            result = import_exports(args.paths, args.out, SystemClock())
            dupes = sum(result.duplicates.values())
            print(
                f"{args.out}: {result.total} rows from {result.files} exports "
                f"({dupes} already present)"
            )
            return 0
        if args.ledger_command == "export":
            if not args.db.exists():
                raise AugurError(f"no ledger at {args.db}")
            counts = export_jsonl(args.db, args.out)
            print(f"{args.out}: {sum(counts.values())} rows from {len(counts)} tables")
            return 0
        return _ledger_check(args.paths)
    except AugurError as exc:
        print(f"augur: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
