"""Every measured number in the README must be one that RESULTS.md states.

RESULTS.md is where each number has its command, model, date, cost and raw
output; the README only repeats. A measured number here is a decimal, a
percentage or an "N of M" count anywhere outside code blocks, compared as
written (so a README cannot round 0.405 to 0.41 or invent a count).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NUMBER = re.compile(
    r"(?<![\w.])[-+−]?\d[\d,]*\.\d+%?(?![\w.])"  # 0.142, −0.016, 99.25%
    r"|(?<![\w.])\d[\d,]*%"  # 47%
    r"|(?<![\w.])\d[\d,]* of \d[\d,]*(?![\w.])"  # 2 of 59
)

# Decimals that are names, not measurements.
NOT_MEASURED = {
    "3.12",  # the Python version
    "3.14",  # the Python version the baseline extra does not support
    "0.1.0",  # the package version
}


def _normal(token: str) -> str:
    return token.replace("−", "-").replace("+", "").replace(",", "")


def _prose(text: str) -> str:
    """The text without fenced code blocks (demo output, commands, the diagram)."""
    kept: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            kept.append(line)
    return "\n".join(kept)


def numbers(text: str) -> set[str]:
    return {_normal(m.group(0)) for m in NUMBER.finditer(text)} - NOT_MEASURED


def test_the_pattern_finds_measurements_and_skips_names() -> None:
    found = numbers("Brier 0.142 [−0.016, +0.017], 47%, 2 of 59, v0.1.0, M10, Python 3.12")
    assert found == {"0.142", "-0.016", "0.017", "47%", "2 of 59"}


def test_every_number_in_the_readme_is_in_results() -> None:
    readme = numbers(_prose((ROOT / "README.md").read_text(encoding="utf-8")))
    results = numbers((ROOT / "RESULTS.md").read_text(encoding="utf-8"))
    missing = sorted(readme - results)
    assert not missing, (
        f"README.md states {missing}, which RESULTS.md does not; quote RESULTS.md as written "
        "or add the measurement there with its command"
    )
