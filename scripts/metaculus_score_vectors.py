"""Score test cases with Metaculus's own scoring code, pinned, for Augur's tests to match.

    uv run --with numpy --with scipy python scripts/metaculus_score_vectors.py \
        tests/fixtures/metaculus_score_vectors.json

Downloads ``scoring/score_math.py`` from github.com/Metaculus/metaculus (BSD
2-Clause) at a pinned commit, replaces its Django imports with small stand-ins
(it reads only ``start_time``, ``end_time`` and ``get_pmf()`` from a forecast),
and runs its spot baseline and spot peer functions on a fixed set of cases:
binary, multiple choice, numeric with closed and open bounds (inside and
beyond the range) and discrete. The cases and Metaculus's scores are written
to JSON; ``tests/test_score_vectors.py`` checks Augur's scores against them
offline. Nothing from Metaculus's code is committed, only its outputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

COMMIT = "8d7ac79b4d12e39b704506c269be9083b0ce8d39"
URL = f"https://raw.githubusercontent.com/Metaculus/metaculus/{COMMIT}/scoring/score_math.py"


def _stub(name: str, **attrs: Any) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Anything:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def load_score_math() -> tuple[types.ModuleType, str]:
    request = urllib.request.Request(URL, headers={"User-Agent": "augur-forecast"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        source = response.read().decode("utf-8")
    for name in ("questions", "scoring", "utils", "utils.the_math"):
        _stub(name)
    _stub(
        "questions.models",
        Forecast=_Anything,
        AggregateForecast=_Anything,
        Question=_Anything,
        QUESTION_CONTINUOUS_TYPES=["numeric", "discrete", "date"],
    )
    _stub("questions.types", AggregationMethod=_Anything)
    _stub("scoring.constants", ScoreTypes=_Anything)
    _stub("scoring.models", Score=_Anything)
    _stub("utils.the_math.aggregations", get_aggregation_history=lambda *a, **k: [])
    _stub("utils.the_math.formulas", string_location_to_bucket_index=lambda *a, **k: None)
    module = types.ModuleType("score_math")
    sys.modules["score_math"] = module
    exec(compile(source, "score_math.py", "exec"), module.__dict__)  # noqa: S102 - pinned source
    return module, hashlib.sha256(source.encode()).hexdigest()


@dataclass
class _Forecast:
    pmf: list[float]
    start_time: datetime
    end_time: datetime | None = None

    def get_pmf(self) -> Any:
        import numpy as np  # type: ignore[import-not-found, unused-ignore]  # run with uv --with numpy

        return np.array(self.pmf, dtype=float)


def _pmf_from_cdf(cdf: list[float]) -> list[float]:
    return [cdf[0], *[b - a for a, b in pairwise(cdf)], 1 - cdf[-1]]


def _ramp(n: int, low: float, high: float, bend: float) -> list[float]:
    """A valid CDF over ``n + 1`` points: from ``low`` to ``high``, skewed by ``bend``."""
    return [low + (high - low) * (i / n) ** bend for i in range(n + 1)]


def cases() -> list[dict[str, Any]]:
    """Each case: type, the forecasts' PMFs, the outcome bucket and the scaling facts."""
    numeric_closed = [_pmf_from_cdf(_ramp(200, 0.0, 1.0, b)) for b in (1.0, 0.7, 1.6)]
    numeric_open = [
        _pmf_from_cdf(_ramp(200, lo, hi, b))
        for lo, hi, b in ((0.05, 0.95, 1.0), (0.12, 0.97, 0.8), (0.02, 0.80, 1.3))
    ]
    discrete = [_pmf_from_cdf(_ramp(11, 0.0, 1.0, b)) for b in (1.0, 0.6, 1.5)]
    return [
        {
            "name": "binary yes",
            "type": "binary",
            "pmfs": [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]],
            "bucket": 1,
            "open_bounds": 0,
        },
        {
            "name": "binary no",
            "type": "binary",
            "pmfs": [[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]],
            "bucket": 0,
            "open_bounds": 0,
        },
        {
            "name": "multiple choice",
            "type": "multiple_choice",
            "pmfs": [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6], [1 / 3, 1 / 3, 1 / 3]],
            "bucket": 0,
            "open_bounds": 0,
        },
        {
            "name": "numeric, closed, inside",
            "type": "numeric",
            "pmfs": numeric_closed,
            "bucket": 74,
            "open_bounds": 0,
        },
        {
            "name": "numeric, open, inside",
            "type": "numeric",
            "pmfs": numeric_open,
            "bucket": 150,
            "open_bounds": 2,
        },
        {
            "name": "numeric, open, below the range",
            "type": "numeric",
            "pmfs": numeric_open,
            "bucket": 0,
            "open_bounds": 2,
        },
        {
            "name": "numeric, open, above the range",
            "type": "numeric",
            "pmfs": numeric_open,
            "bucket": 201,
            "open_bounds": 2,
        },
        {
            "name": "discrete, closed",
            "type": "discrete",
            "pmfs": discrete,
            "bucket": 4,
            "open_bounds": 0,
        },
    ]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    sm, digest = load_score_math()
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    spot = datetime(2026, 1, 2, tzinfo=UTC).timestamp()
    out: list[dict[str, Any]] = []
    for case in cases():
        forecasts = [_Forecast(p, t0) for p in case["pmfs"]]
        baseline = sm.evaluate_forecasts_baseline_spot_forecast(
            forecasts, case["bucket"], spot, case["type"], case["open_bounds"]
        )
        peer = sm.evaluate_forecasts_peer_spot_forecast(
            forecasts, None, case["bucket"], spot, case["type"]
        )
        out.append(
            case
            | {
                "metaculus_spot_baseline": [float(s.score) for s in baseline],
                "metaculus_spot_peer": [float(s.score) for s in peer],
            }
        )
    payload = {
        "source": URL,
        "commit": COMMIT,
        "sha256": digest,
        "licence": "Metaculus, BSD 2-Clause (outputs only)",
        "cases": out,
    }
    Path(argv[0]).write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases scored by score_math.py @ {COMMIT[:7]} to {argv[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
