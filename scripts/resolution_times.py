"""When did each C4 market's outcome become public? Ask the market itself.

    python scripts/resolution_times.py data/calibration/m8-resolution-times.json

ForecastBench's ``resolution_date`` is not always the day an outcome became
known: for a market that closed before it resolved it is often the close
date, weeks earlier (M8's review found 155 of 438 Polymarket and Manifold
markets resolved after it). Q4's harness calibrates a question only on
outcomes known before it opened, so it needs the real time. This script
reads it from each market's public API:

* Manifold: ``resolutionTime`` (``/v0/market/<id>``);
* Polymarket: ``closedTime`` (the gamma API, by condition id), the later of
  it and ``umaEndDate`` where both are present;
* Metaculus: the post's ``actual_resolve_time`` (needs ``METACULUS_TOKEN``,
  which is read from the environment and never printed);
* INFER: no public API; recorded with no time, so the harness never trains
  on it.

Standard library only; one request at a time with a pause, as the APIs ask.
Run it again to retry only the markets an earlier run could not date.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from augur.forecastbench import load_set
from augur.q4 import FORECASTBENCH, RELEASED, SETS

AGENT = "augur-forecast (research)"


def _get(url: str, headers: dict[str, str] | None = None) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT, **(headers or {})})  # noqa: S310
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    return None


def _iso(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().replace(" ", "T")
    if text.endswith("+00"):
        text += ":00"
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def known_at(source: str, market: str, token: str | None) -> tuple[str | None, str]:
    """The instant the outcome was public, and the field that says so."""
    if source == "manifold":
        d = _get(f"https://api.manifold.markets/v0/market/{market}")
        t = d.get("resolutionTime") if isinstance(d, dict) else None
        if not t:
            return None, "manifold: no resolutionTime"
        return datetime.fromtimestamp(t / 1000, UTC).isoformat(), "manifold resolutionTime"
    if source == "polymarket":
        rows = _get(f"https://gamma-api.polymarket.com/markets?condition_ids={market}&closed=true")
        row = rows[0] if isinstance(rows, list) and rows else {}
        times = [t for t in (_iso(row.get("closedTime")), _iso(row.get("umaEndDate"))) if t]
        if not times:
            return None, "polymarket: no closedTime or umaEndDate"
        return max(times), "polymarket closedTime (or umaEndDate if later)"
    if source == "metaculus":
        if not token:
            return None, "metaculus: no METACULUS_TOKEN"
        d = _get(
            f"https://www.metaculus.com/api/posts/{market}/",
            {"Authorization": f"Token {token}"},
        )
        q = d.get("question") or {} if isinstance(d, dict) else {}
        t = _iso(q.get("actual_resolve_time"))
        if not t:
            return None, "metaculus: no actual_resolve_time"
        return t, "metaculus actual_resolve_time"
    return None, f"{source}: no public API for resolution times"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(argv[0])
    token = os.environ.get("METACULUS_TOKEN") or None
    markets: dict[str, tuple[str, str]] = {}
    for qs in SETS:
        report = load_set(
            FORECASTBENCH / f"{qs}-llm.subset.json",
            FORECASTBENCH / f"{qs}_resolution_set.subset.json",
            released=RELEASED,
        )
        for q in report.kept:
            markets.setdefault(f"{q.source}:{q.fb_id}", (q.source, q.fb_id))
    # A second run keeps what an earlier one found and asks again only for the rest
    # (Metaculus answers some requests with 429 when they come in quick succession).
    earlier: dict[str, dict[str, str | None]] = (
        json.loads(out.read_text(encoding="utf-8"))["markets"] if out.exists() else {}
    )
    result: dict[str, dict[str, str | None]] = {}
    for key, (source, market) in sorted(markets.items()):
        if earlier.get(key, {}).get("known_at"):
            result[key] = earlier[key]
            continue
        try:
            at, evidence = known_at(source, market, token)
        except (OSError, ValueError) as exc:
            at, evidence = None, f"{source}: request failed ({type(exc).__name__})"
        result[key] = {"known_at": at, "evidence": evidence}
        time.sleep(1.0 if source == "metaculus" else 0.25)
    found = sum(1 for r in result.values() if r["known_at"])
    payload = {
        "fetched": date.today().isoformat(),  # noqa: DTZ011 - the day of the fetch, for the record
        "markets": result,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{out}: {found} of {len(result)} markets with a resolution time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
