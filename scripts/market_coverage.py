"""How often does a live MiniBench question have a similar market? (M5, P5.)

    METACULUS_TOKEN=... uv run python scripts/market_coverage.py \\
        data/runs/m3-question-windows.json minibench data/runs/m5-market-coverage.json

Reads the post ids of one round from the M3 window measurement, fetches each
question from the Metaculus API, and runs the ``markets`` searcher's
similar-market search as of the question's open time (when the bot would have
forecast it). Writes every hit with its price and the timestamp that dates
it. Whether the top hit is about the same event is judged by hand afterwards,
in ``data/runs/m5-market-coverage-labels.json``; the script never guesses.
Read-only and free: public market APIs and Metaculus reads.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from augur.clock import SystemClock, to_iso
from augur.config import Settings
from augur.metaculus import MetaculusClient
from augur.research.markets import MarketsSearcher

LIMIT = 3


async def _run(window_file: Path, round_name: str) -> dict[str, Any]:
    windows = json.loads(window_file.read_text(encoding="utf-8"))
    posts = [q for q in windows["questions"] if q["round"] == round_name]
    settings = Settings.from_env(os.environ)
    clock = SystemClock()
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=60.0) as http:
        metaculus = MetaculusClient.from_settings(settings, http)
        markets = MarketsSearcher(http, clock, metaculus=metaculus)
        for entry in posts:
            q, _raw = await metaculus.get_question(int(entry["post_id"]))
            as_of = q.open_time
            sources = await markets.search_question(q, as_of, LIMIT)
            hits = [
                {
                    "platform": s.quote.platform,
                    "match": s.quote.match,
                    "title": s.quote.title,
                    "url": s.quote.url,
                    "probability": s.quote.probability,
                    "price_time": to_iso(s.quote.price_time),
                }
                for s in sources
                if s.quote is not None
            ]
            rows.append(
                {
                    "post_id": q.post_id,
                    "question_id": q.question_id,
                    "type": q.type,
                    "title": q.title,
                    "as_of": to_iso(as_of),
                    "hits": hits,
                }
            )
            print(f"{q.post_id} {q.type:15} {len(hits)} hit(s)  {q.title[:70]}")
    searched = [h for r in rows for h in r["hits"] if h["match"] == "search"]
    return {
        "command": "uv run python scripts/market_coverage.py "
        f"{window_file} {round_name} OUT",
        "fetched_at": to_iso(clock.now()),
        "round": round_name,
        "questions": len(rows),
        "with_a_search_hit": sum(any(h["match"] == "search" for h in r["hits"]) for r in rows),
        "with_community_prior": sum(
            any(h["platform"] == "metaculus" for h in r["hits"]) for r in rows
        ),
        "search_hits": len(searched),
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    result = asyncio.run(_run(Path(argv[0]), argv[1]))
    Path(argv[2]).write_text(
        json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{result['questions']} questions; {result['with_a_search_hit']} with a priced "
        f"similar market; {result['with_community_prior']} with a visible community prior"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
