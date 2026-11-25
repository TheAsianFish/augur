"""Re-reading the one-round floor rule: a synthetic simulation (M14, item 9).

The floor rule reads one MiniBench round. If the same candidate is read
again after each new round, or each round is read alone, the chance that
the rule admits a candidate with no lead at least once grows with the
number of looks. This module measures how much, on made-up data.

Each simulated season draws ``ROUNDS`` rounds of n head-to-head values
from a normal with mean 0 (a candidate with no lead) and SD ``SPREAD``;
S = 0, so the rule's two conditions coincide. The rule admits when the
lower end of the 90% interval, read as the normal approximation to the
bootstrap, ``mean - Z * sample_SD / sqrt(m)``, is above 0 (m is the number
of questions read; the rule's minimum of 20 is always met here). For each
k = 1 to 4 it records the share of seasons in which the rule admits at
least once within the first k rounds, (a) applied after each round j <= k
to all questions of rounds 1 to j pooled (cumulative looks) and (b)
applied to each round j <= k alone.

One ``random.Random(SEED)`` is made per round size, so each size's table
is reproducible alone. Per-round sums and sums of squares are kept so the
pooled mean and SD cost nothing extra. Nothing here is real data: every
printed form carries the label "synthetic". For independent rounds with
exact nominal coverage the round-by-round rate is
``analytic_separate(k) = 1 - (1 - 0.05) ** k``, which the simulated
separate column should sit beside.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass

ROUND_SIZES = (30, 60)
ROUNDS = 4
SEASONS = 20_000
SEED = 20261125
SPREAD = 32.0
NOMINAL = 0.05
Z = statistics.NormalDist().inv_cdf(0.95)


@dataclass(frozen=True)
class Table:
    """Shares of seasons with at least one admission, by rounds seen k = 1..ROUNDS."""

    n: int
    cumulative: tuple[float, ...]
    separate: tuple[float, ...]


def analytic_separate(k: int, nominal: float = NOMINAL) -> float:
    """Round-by-round rate for k independent rounds at exact nominal coverage."""
    return 1.0 - (1.0 - nominal) ** k


def admits(total: float, total_sq: float, m: int, z: float = Z) -> bool:
    """The rule on m questions with the given sum and sum of squares."""
    mean = total / m
    var = max((total_sq - total * total / m) / (m - 1), 0.0)
    return mean - z * math.sqrt(var / m) > 0.0


def simulate(
    round_sizes: tuple[int, ...] = ROUND_SIZES,
    rounds: int = ROUNDS,
    seasons: int = SEASONS,
    seed: int = SEED,
    spread: float = SPREAD,
) -> list[Table]:
    """One table per round size."""
    tables: list[Table] = []
    for n in round_sizes:
        rng = random.Random(seed)
        gauss = rng.gauss
        cum_first = [0] * rounds
        sep_first = [0] * rounds
        for _ in range(seasons):
            total = 0.0
            total_sq = 0.0
            cum_done = False
            sep_done = False
            for j in range(rounds):
                values = [gauss(0.0, spread) for _ in range(n)]
                s = math.fsum(values)
                ss = math.fsum(v * v for v in values)
                total += s
                total_sq += ss
                if not sep_done and admits(s, ss, n):
                    sep_done = True
                if not cum_done and admits(total, total_sq, n * (j + 1)):
                    cum_done = True
                if cum_done:
                    cum_first[j] += 1
                if sep_done:
                    sep_first[j] += 1
        tables.append(
            Table(
                n=n,
                cumulative=tuple(c / seasons for c in cum_first),
                separate=tuple(c / seasons for c in sep_first),
            )
        )
    return tables


def snapshot() -> dict[str, object]:
    """The registered simulation as a JSON-serialisable dict."""
    tables = simulate()
    return {
        "label": "synthetic",
        "method": (
            "no-lead normal draws, S = 0, rule lower end = mean - z*SD/sqrt(m); "
            "share of seasons admitted at least once in the first k rounds"
        ),
        "seasons": SEASONS,
        "seed": SEED,
        "spread": SPREAD,
        "rounds": ROUNDS,
        "nominal": NOMINAL,
        "z": round(Z, 6),
        "tables": [
            {
                "n": t.n,
                "cumulative": [round(x, 4) for x in t.cumulative],
                "separate": [round(x, 4) for x in t.separate],
            }
            for t in tables
        ],
    }


def snapshot_text(snap: dict[str, object]) -> str:
    """Byte-stable text of a snapshot."""
    return json.dumps(snap, sort_keys=True, indent=1) + "\n"
