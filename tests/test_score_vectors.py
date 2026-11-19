"""Augur's scores against Metaculus's own scoring code, on fixed cases.

``tests/fixtures/metaculus_score_vectors.json`` was written by
``scripts/metaculus_score_vectors.py``, which ran Metaculus's
``scoring/score_math.py`` (pinned commit) on each case: the spot baseline
score and the spot peer score of every forecaster. Augur must agree to
1e-9 on every type: binary, multiple choice, numeric with closed and open
bounds (inside and beyond the range) and discrete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from augur.distributions import Scaling
from augur.score import baseline_score, peer_score

VECTORS = Path(__file__).parent / "fixtures" / "metaculus_score_vectors.json"
DATA = json.loads(VECTORS.read_text(encoding="utf-8"))


def _scaling(case: dict[str, Any]) -> Scaling | None:
    if case["type"] not in ("numeric", "discrete"):
        return None
    open_bounds = case["open_bounds"] > 0
    return Scaling(
        range_min=0.0,
        range_max=1.0,
        open_lower_bound=open_bounds,
        open_upper_bound=open_bounds,
        inbound_outcome_count=len(case["pmfs"][0]) - 2,
    )


@pytest.mark.parametrize("case", DATA["cases"], ids=[c["name"] for c in DATA["cases"]])
def test_augur_matches_metaculus_score_math(case: dict[str, Any]) -> None:
    bucket = case["bucket"]
    qtype = case["type"]
    scaling = _scaling(case)
    ps = [pmf[bucket] for pmf in case["pmfs"]]
    for i, p in enumerate(ps):
        ours = baseline_score(
            qtype,
            p,
            option_count=len(case["pmfs"][i]),
            bucket=bucket,
            scaling=scaling,
        )
        assert ours == pytest.approx(case["metaculus_spot_baseline"][i], abs=1e-9)
        peer = peer_score(qtype, p, [q for j, q in enumerate(ps) if j != i])
        assert peer == pytest.approx(case["metaculus_spot_peer"][i], abs=1e-9)


def test_the_vectors_cover_every_type_and_both_tails() -> None:
    names = {c["name"] for c in DATA["cases"]}
    assert {c["type"] for c in DATA["cases"]} == {
        "binary",
        "multiple_choice",
        "numeric",
        "discrete",
    }
    assert {"numeric, open, below the range", "numeric, open, above the range"} <= names
    assert DATA["commit"].startswith("8d7ac79")
