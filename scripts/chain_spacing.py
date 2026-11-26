#!/usr/bin/env python3
"""How far apart the Tournament workflow's ticks really start. Standard library only.

With ``AUGUR_CHAIN_MINUTES`` set, each run on ``main`` dispatches the next one
itself (``workflow_dispatch`` with ``chained``), and the cron restarts the chain
if it breaks. This reads a saved list of the workflow's runs, from either

    RUNS="repos/$REPO/actions/workflows/tournament.yml/runs?per_page=100"
    gh api "$RUNS" > runs.json
    curl -s "https://api.github.com/$RUNS" > runs.json

(the repository is public, so the second needs no token), keeps the runs on
``main`` started by ``schedule`` or ``workflow_dispatch`` since ``--since``, and
prints the gap between each pair of consecutive starts, their median and
largest, and how many fall within ``--tolerance`` minutes of ``--target``.

    python3 scripts/chain_spacing.py runs.json --since 2026-11-26T00:00:00Z --target 30

It does not tell a chained dispatch from a person's: dispatch the two
bot-testing-area checks before ``--since``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENTS = ("schedule", "workflow_dispatch")


def _when(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def starts(
    runs: dict[str, Any], since: datetime, branch: str = "main", workflow: str = "Tournament"
) -> list[datetime]:
    """Start instants of the kept runs, oldest first."""
    out: list[datetime] = []
    for run in runs.get("workflow_runs", []):
        if run.get("event") not in EVENTS or run.get("head_branch") != branch:
            continue
        if run.get("name") != workflow:
            continue
        at = _when(str(run.get("run_started_at") or run["created_at"]))
        if at >= since:
            out.append(at)
    return sorted(out)


def spacing(times: list[datetime]) -> list[float]:
    """Minutes between consecutive starts."""
    return [(b - a).total_seconds() / 60 for a, b in itertools.pairwise(times)]


def report(gaps: list[float], target: float, tolerance: float) -> list[str]:
    if not gaps:
        return ["fewer than two runs since the start: no spacing to measure"]
    near = sum(1 for g in gaps if abs(g - target) <= tolerance)
    return [
        f"runs: {len(gaps) + 1}; gaps: {len(gaps)}",
        f"median gap {statistics.median(gaps):.1f} min; largest {max(gaps):.1f} min; "
        f"smallest {min(gaps):.1f} min",
        f"within {tolerance:g} min of the {target:g}-minute target: {near} of {len(gaps)}",
        "gaps (min): " + ", ".join(f"{g:.1f}" for g in gaps),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, help="the saved runs JSON")
    parser.add_argument("--since", required=True, help="ISO 8601 UTC; earlier runs are ignored")
    parser.add_argument("--target", type=float, default=30.0, help="AUGUR_CHAIN_MINUTES")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--workflow", default="Tournament", help="the workflow's name")
    args = parser.parse_args(argv)
    try:
        runs = json.loads(args.runs.read_text(encoding="utf-8"))
        since = _when(args.since)
    except (OSError, ValueError) as exc:
        print(f"chain_spacing: {exc}", file=sys.stderr)
        return 1
    print(
        "\n".join(
            report(
                spacing(starts(runs, since, args.branch, args.workflow)),
                args.target,
                args.tolerance,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
