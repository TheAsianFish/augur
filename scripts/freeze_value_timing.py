"""When was ForecastBench's Polymarket "freeze value" read? (M5, P5.)

    uv run python scripts/freeze_value_timing.py data/forecastbench/2026-06-21 \\
        data/ledgers/m5-replay-c2-prices.sqlite > data/runs/m5-freeze-value-timing.txt

For each Polymarket question whose recorded as-of price (``augur backtest
prices``, the last history point at or before the freeze instant) differs from
ForecastBench's ``freeze_datetime_value`` by more than 0.01, reads the hourly
price history from three days before to three days after the freeze and
prints the hour offsets at which the history equals ForecastBench's value
exactly. A positive offset means the published value is a price from after
the freeze instant. Read-only and free: Polymarket's public APIs.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from augur.forecastbench import load_set
from augur.replay import recorded_prices

GAMMA = "https://gamma-api.polymarket.com/markets"
CLOB = "https://clob.polymarket.com/prices-history"
WINDOW_S = 3 * 86_400


def _get(url: str, params: dict[str, str]) -> Any:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(  # noqa: S310 - fixed https hosts
        full, headers={"User-Agent": "augur-forecast (research)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https hosts
        return json.load(response)


def _yes_token(condition_id: str) -> str | None:
    for closed in ("true", "false"):
        found = _get(GAMMA, {"condition_ids": condition_id, "closed": closed})
        if found:
            return str(json.loads(found[0]["clobTokenIds"])[0])
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    stem, ledger = argv[0], Path(argv[1])
    loaded = load_set(
        Path(f"{stem}-llm.subset.json"),
        Path(f"{stem}_resolution_set.subset.json"),
        released=date.min,
    )
    prices = recorded_prices(ledger, f"forecastbench:{loaded.question_set}")
    rows = []
    for fb in loaded.kept:
        source = prices.get(fb.question_id)
        if fb.source != "polymarket" or source is None or source.quote is None:
            continue
        if fb.freeze_value is None or abs(source.quote.probability - fb.freeze_value) <= 0.01:
            continue
        token = _yes_token(fb.fb_id)
        if token is None:
            continue
        freeze = int(fb.freeze_datetime.timestamp())
        history = _get(
            CLOB,
            {
                "market": token,
                "startTs": str(freeze - WINDOW_S),
                "endTs": str(freeze + WINDOW_S),
                "fidelity": "60",
            },
        )["history"]
        offsets = sorted(
            (p["t"] - freeze) / 3600 for p in history if abs(p["p"] - fb.freeze_value) < 1e-9
        )
        rows.append((fb.fb_id, source.quote.probability, fb.freeze_value, offsets))
    print(f"{loaded.question_set}: {len(rows)} Polymarket questions differ by more than 0.01")
    print("condition id  as-of price  freeze value  hours after the freeze where they are equal")
    after = 0
    for fb_id, ours, theirs, offsets in rows:
        shown = ", ".join(f"{o:+.1f}" for o in offsets[:6]) or "none in +/-3 days"
        after += any(o > 0 for o in offsets)
        print(f"{fb_id[:12]}  {ours:.4f}       {theirs:.4f}        {shown}")
    print(f"{after} of {len(rows)} equal a price from after the freeze instant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
