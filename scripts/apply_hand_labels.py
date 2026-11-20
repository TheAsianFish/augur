"""Install a filled-in label file as the judge's hand labels.

``scripts/judge_sample.py`` writes the 40 sources to score with ``score``
left null. Once a person has filled every score (0 to 4 on the rubric in
augur/judge.py), this checks the file names the same sources in the same
order as the committed reference labels, that every score is an integer
0 to 4, and writes it over the reference file with ``by_hand`` set and the
rater named. ``make results`` then reports P9 as held or failed.

    uv run python scripts/apply_hand_labels.py /tmp/labels.json --rater "the maintainer"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REFERENCE = Path("data/judge/m6-reference-labels.json")


class LabelError(ValueError):
    """The filled file cannot replace the reference labels; the message says why."""


def _key(label: dict[str, Any]) -> tuple[str, int, str]:
    return (str(label["arm"]), int(label["question_id"]), str(label["source_id"]))


def merge(filled: dict[str, Any], reference: dict[str, Any], rater: str) -> dict[str, Any]:
    """The reference payload with the filled scores, marked as hand labels."""
    if not rater.strip():
        raise LabelError("name the rater with --rater")
    new: list[dict[str, Any]] = filled.get("labels", [])
    old: list[dict[str, Any]] = reference.get("labels", [])
    if [_key(x) for x in new] != [_key(x) for x in old]:
        raise LabelError(
            "the filled file does not list the reference's sources in the same order; "
            "draw it again with scripts/judge_sample.py --per-arm 20"
        )
    for label in new:
        score = label.get("score")
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise LabelError(
                f"{label['arm']} {label['source_id']}: score must be an integer 0 to 4, "
                f"got {score!r}"
            )
    return {
        "rubric": reference.get("rubric", "El Lahib et al. 0-4 (augur/judge.py)"),
        "sample": reference.get("sample", ""),
        "rater": f"By hand: {rater.strip()}, from the sheet scripts/judge_sample.py writes, "
        "blind to the judge's scores and to the earlier provisional rating.",
        "by_hand": True,
        "labels": [
            {
                "arm": x["arm"],
                "question_id": x["question_id"],
                "source_id": x["source_id"],
                "score": x["score"],
                "note": str(x.get("note", "")),
            }
            for x in new
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("filled", type=Path)
    ap.add_argument("--rater", required=True)
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    filled = json.loads(args.filled.read_text(encoding="utf-8"))
    try:
        payload = merge(filled, reference, args.rater)
    except LabelError as e:
        print(f"error: {e}")
        return 1
    args.reference.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(payload['labels'])} hand labels written to {args.reference}; run make results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
