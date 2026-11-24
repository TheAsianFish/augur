"""M12's registration: RESULTS.md carries PLAN.md's registered text byte for byte."""

from __future__ import annotations

import re
from pathlib import Path

from augur.report import render_m12_registration

REPO = Path(__file__).resolve().parents[1]


def test_registration_is_planned_text_verbatim() -> None:
    plan = (REPO / "PLAN.md").read_text(encoding="utf-8")
    start, end = "<!-- m12-registration:begin -->\n", "<!-- m12-registration:end -->"
    want = plan.split(start, 1)[1].split(end, 1)[0]
    assert render_m12_registration(REPO) == want
    results = (REPO / "RESULTS.md").read_text(encoding="utf-8")
    m = re.search(
        r"<!-- augur:begin m12-registration -->\n(.*?)<!-- augur:end m12-registration -->",
        results,
        re.S,
    )
    assert m is not None
    assert m.group(1) == want


def test_registration_names_the_fixed_numbers() -> None:
    body = render_m12_registration(REPO)
    for needle in ("one-third Augur", "fewer than 20", "fewer than 100", "00:00 UTC of D + 1"):
        assert needle in body
    assert [f"P{n}" for n in range(34, 41)] == re.findall(r"^\| (P\d+) \|", body, re.M)
