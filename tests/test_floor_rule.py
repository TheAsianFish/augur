"""M11's floor rule: RESULTS.md carries PLAN.md's registered text byte for byte."""

from __future__ import annotations

import re
from pathlib import Path

from augur.report import render_floor_rule

REPO = Path(__file__).resolve().parents[1]


def test_floor_rule_is_planned_text_verbatim() -> None:
    plan = (REPO / "PLAN.md").read_text(encoding="utf-8")
    start, end = "<!-- floor-rule:begin -->\n", "<!-- floor-rule:end -->"
    want = plan.split(start, 1)[1].split(end, 1)[0]
    assert render_floor_rule(REPO) == want
    results = (REPO / "RESULTS.md").read_text(encoding="utf-8")
    m = re.search(
        r"<!-- augur:begin floor-rule -->\n(.*?)<!-- augur:end floor-rule -->", results, re.S
    )
    assert m is not None
    assert m.group(1) == want


def test_floor_rule_names_the_registered_numbers() -> None:
    body = render_floor_rule(REPO)
    for needle in ("90%", "10,000 resamples", "seed 20261119", "n is below 20", "2026-12-13"):
        assert needle in body
