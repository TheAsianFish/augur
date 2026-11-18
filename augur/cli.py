"""The ``augur`` command line."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from augur.clock import SystemClock
from augur.config import Settings
from augur.errors import AugurError, MissingCredentialsError
from augur.leakage import scan_ledger
from augur.ledger import Ledger
from augur.llm.factory import build_provider, default_http_client
from augur.log import configure_logging, get_logger
from augur.metaculus import MetaculusClient
from augur.research.asknews import AskNewsSearcher
from augur.research.source import Searcher
from augur.research.wikipedia import WikipediaSearcher
from augur.runner import TOURNAMENTS, ForecastRun

log = get_logger("augur.cli")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="augur", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    fc = sub.add_parser(
        "forecast",
        help="forecast open questions (a dry run unless --submit)",
        description=(
            "Forecast a tournament's open binary questions. Without METACULUS_TOKEN it runs "
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

    led = sub.add_parser("ledger", help="inspect a ledger")
    led_sub = led.add_subparsers(dest="ledger_command", required=True)
    check = led_sub.add_parser("check", help="scan a ledger for leakage; exit 1 on a violation")
    check.add_argument("paths", type=Path, nargs="+")
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
        provider = build_provider(settings, http)
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
                provider=provider,
                searchers=searchers,
                client=client,
                out=sys.stdout,
            )
            try:
                report = await run.run(
                    args.tournament,
                    submit=args.submit,
                    samples=args.samples,
                    max_questions=max_questions,
                )
            finally:
                await provider.aclose()
        print(f"ledger: {ledger_path} (run {report.run_id})")
        return 1 if report.count("failed") else 0


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
        return _ledger_check(args.paths)
    except AugurError as exc:
        print(f"augur: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
