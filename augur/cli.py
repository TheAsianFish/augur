"""The ``augur`` command line."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from augur.clock import SystemClock
from augur.config import Settings
from augur.errors import AugurError, ConfigError, MissingCredentialsError
from augur.export import export_jsonl
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
from augur.research.asknews import AskNewsSearcher
from augur.research.source import Searcher
from augur.research.wikipedia import WikipediaSearcher
from augur.runner import TOURNAMENTS, ForecastRun, format_probe

log = get_logger("augur.cli")


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
    fc.add_argument("--submit", action="store_true", help="post forecasts and private comments")
    fc.add_argument("--samples", type=Path, help="forecast sample posts from this directory")
    fc.add_argument("--max-questions", type=int, help="override AUGUR_MAX_QUESTIONS_PER_RUN")
    fc.add_argument("--ledger", type=Path, help="override AUGUR_LEDGER")
    fc.add_argument("--no-asknews", action="store_true", help="skip AskNews even with keys")
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

    led = sub.add_parser("ledger", help="inspect a ledger")
    led_sub = led.add_subparsers(dest="ledger_command", required=True)
    check = led_sub.add_parser("check", help="scan a ledger for leakage; exit 1 on a violation")
    check.add_argument("paths", type=Path, nargs="+")
    export = led_sub.add_parser(
        "export", help="write a ledger as JSONL, one row per line, byte-stable"
    )
    export.add_argument("db", type=Path)
    export.add_argument("out", type=Path)
    return parser


async def _forecast(args: argparse.Namespace, settings: Settings) -> int:
    clock = SystemClock()
    ledger_path: Path = args.ledger or settings.ledger_path
    max_questions = (
        args.max_questions if args.max_questions is not None else settings.max_questions_per_run
    )
    if args.submit and not settings.has_metaculus_token:
        raise MissingCredentialsError("--submit", ("METACULUS_TOKEN",))
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
        if client is None:
            print("METACULUS_TOKEN is not set: dry run on the committed sample questions.")
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
            )
            try:
                report = await run.run(
                    args.tournament,
                    submit=args.submit,
                    samples=args.samples,
                    max_questions=max_questions,
                    reforecast=args.reforecast,
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
    args = _parser().parse_args(argv)
    try:
        settings = Settings.from_env(os.environ)
        configure_logging(settings.log_level)
        if args.command == "forecast":
            return asyncio.run(_forecast(args, settings))
        if args.command == "models":
            return asyncio.run(_models(args, settings))
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
