"""What a token can see of resolved questions: resolution, aggregates, its own scores.

    METACULUS_TOKEN=... python3 scripts/resolution_visibility.py [TOURNAMENT ...]

For each tournament (default: two MiniBench rounds, Fall 2026 FutureEval and
Spring 2026), lists up to 20 resolved posts and counts how many show a
``resolution``, a community aggregate (``latest`` or ``score_data``) and a
``my_forecasts.score_data``. Standard library only; read-only; one request
per tournament page, one per post. Prints one line per tournament.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

API = "https://www.metaculus.com/api"
DEFAULT = ["minibench-2026-11-02", "minibench-2026-10-19", "33121", "spring-aib-2026"]
TYPES = "binary,multiple_choice,numeric,discrete"


def _get(path: str, token: str, params: dict[str, str] | None = None) -> Any:
    url = f"{API}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        url, headers={"Authorization": f"Token {token}", "User-Agent": "augur-forecast"}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = float(exc.headers.get("Retry-After") or 10 * (attempt + 1))
            time.sleep(min(wait, 120.0))
    raise RuntimeError("unreachable")


def main(argv: list[str]) -> int:
    token = os.environ.get("METACULUS_TOKEN", "")
    if not token:
        print("set METACULUS_TOKEN", file=sys.stderr)
        return 2
    me = _get("/users/me/", token)
    print(f"account: {me.get('username')} (bot: {me.get('is_bot')})")
    for tournament in argv or DEFAULT:
        listing = _get(
            "/posts/",
            token,
            {
                "tournaments": tournament,
                "statuses": "resolved",
                "limit": "20",
                "forecast_type": TYPES,
            },
        )
        posts = [p for p in listing.get("results", []) if isinstance(p.get("question"), dict)]
        seen = {"resolution": 0, "aggregate": 0, "my_score": 0}
        types: dict[str, int] = {}
        for post in posts:
            time.sleep(2.0)
            q = _get(f"/posts/{post['id']}/", token)["question"]
            types[q["type"]] = types.get(q["type"], 0) + 1
            if q.get("resolution") not in (None, ""):
                seen["resolution"] += 1
            aggs = q.get("aggregations") or {}
            if any(
                isinstance(a, dict) and (a.get("latest") or a.get("score_data"))
                for a in aggs.values()
            ):
                seen["aggregate"] += 1
            if (q.get("my_forecasts") or {}).get("score_data"):
                seen["my_score"] += 1
        kinds = ", ".join(f"{t} {n}" for t, n in sorted(types.items()))
        print(
            f"{tournament}: {len(posts)} resolved posts ({kinds}); resolution shown on "
            f"{seen['resolution']}, community aggregate on {seen['aggregate']}, "
            f"this account's score on {seen['my_score']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
