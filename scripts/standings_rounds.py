#!/usr/bin/env python3
"""Which MiniBench rounds the daily standings job should re-read. Standard library only.

The current round is always ``minibench``; once a round closes, Metaculus
gives it a dated slug (``minibench-2026-11-02``, its start date) and the
``minibench`` slug moves to the next round. Rounds start every 14 days, so
from the current round's snapshot this prints the current slug and the dated
slugs of the rounds before it, which are the ones that finalize after their
close and whose final leaderboard is the result.

    python3 scripts/standings_rounds.py SNAPSHOT.json [--back N]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROUND_DAYS = 14


def previous_slugs(snapshot: dict[str, object], back: int) -> list[str]:
    """Dated slugs of the ``back`` rounds before the snapshot's round, newest first."""
    round_ = snapshot.get("round")
    if not isinstance(round_, dict):
        raise ValueError("the snapshot has no round record")
    start = round_.get("start_date")
    if not isinstance(start, str):
        raise ValueError("the round record has no start_date")
    began = datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(UTC)
    return [
        f"minibench-{(began - timedelta(days=ROUND_DAYS * k)).date().isoformat()}"
        for k in range(1, back + 1)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--back", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        slugs = previous_slugs(snapshot, args.back)
    except (OSError, ValueError) as exc:
        print(f"standings_rounds: {exc}", file=sys.stderr)
        return 1
    print("\n".join(slugs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
