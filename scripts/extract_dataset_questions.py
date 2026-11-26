"""Cut ForecastBench's question and resolution sets down to their dataset questions (M15).

    python -I scripts/extract_dataset_questions.py RAW_DIR OUT_DIR --fetched 2026-11-26

``RAW_DIR`` holds the files as published, renamed ``<set>-llm.json`` and
``<set>-res.json`` (the question set and its resolution set, from
``forecastingresearch/forecastbench-datasets``, ``datasets/question_sets``
and ``datasets/resolution_sets``). For every 2026 set from :data:`FIRST` to
:data:`LAST`, writes ``OUT_DIR/<set>.json``: each single-series dataset
question (source, id, freeze instant, freeze value, URL) with its outcome at
every resolution date the resolution set resolves to 0 or 1, and the count
of everything left out, by reason. A provenance header gives both files'
URLs, sizes, SHA-256 and the fetch date. The sets from :data:`CORPUS_FIRST`
on are the scored corpus; the earlier ones are history only
(PLAN.md, ``m15-registration``, items 1 and 2).

The files are untrusted data: read only as JSON, standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
LICENCE = "ForecastBench datasets, Forecasting Research Institute, CC BY-SA 4.0"
DATASET_SOURCES = frozenset({"acled", "dbnomics", "fred", "wikipedia", "yfinance"})
MARKET_SOURCES = frozenset({"manifold", "metaculus", "polymarket", "infer", "kalshi"})
FIRST = "2026-01-04"
CORPUS_FIRST = "2026-07-05"
LAST = "2026-09-27"


class ExtractError(ValueError):
    """The input is not what this script expects; nothing is written."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value) if value in (0, 1) else None


def _freeze_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def cut(questions: dict[str, Any], resolutions: dict[str, Any], fetched: str) -> dict[str, Any]:
    """One set's single-series dataset questions with their resolved outcomes.

    ``fetched`` (ISO day) splits the listed dates with no resolved row into
    those still ahead (a horizon not reached) and those already past.
    """
    if questions.get("question_set") != resolutions.get("question_set"):
        raise ExtractError(
            f"the resolution set resolves {resolutions.get('question_set')!r}, "
            f"not {questions.get('question_set')!r}"
        )
    removed: dict[str, int] = {}

    def remove(reason: str, n: int = 1) -> None:
        removed[reason] = removed.get(reason, 0) + n

    resolved: dict[tuple[str, str], dict[str, int]] = {}
    for r in resolutions.get("resolutions", []):
        if str(r.get("source")) not in DATASET_SOURCES or not isinstance(r.get("id"), str):
            continue
        key = (str(r["source"]), str(r["id"]))
        day = str(r.get("resolution_date"))[:10]
        y = _outcome(r.get("resolved_to")) if r.get("resolved") is True else None
        if y is None:
            remove("a dated row not resolved to 0 or 1")
            continue
        if day in resolved.setdefault(key, {}):
            raise ExtractError(f"{key[0]}:{key[1]} is resolved twice at {day}")
        resolved[key][day] = y
    kept: list[dict[str, Any]] = []
    total = 0
    for q in questions.get("questions", []):
        source = str(q.get("source"))
        if source in MARKET_SOURCES:
            continue
        total += 1
        if source not in DATASET_SOURCES:
            remove(f"another dataset source ({source})")
            continue
        if not isinstance(q.get("id"), str):
            remove("a pair question (a list of ids)")
            continue
        key = (source, str(q["id"]))
        outcomes = resolved.pop(key, {})
        listed = [str(d)[:10] for d in q.get("resolution_dates") or []]
        missing = [d for d in listed if d not in outcomes]
        ahead = [d for d in missing if d > fetched]
        if ahead:
            remove("a listed resolution date after the fetch day", len(ahead))
        if len(missing) > len(ahead):
            remove(
                "a listed past resolution date with no row resolved to 0 or 1",
                len(missing) - len(ahead),
            )
        if not outcomes:
            remove("a question with no resolved date")
            continue
        kept.append(
            {
                "source": source,
                "id": key[1],
                "freeze_datetime": str(q.get("freeze_datetime")),
                "freeze_value": _freeze_value(q.get("freeze_datetime_value")),
                "url": str(q.get("url") or ""),
                "resolutions": dict(sorted(outcomes.items())),
            }
        )
    if resolved:
        remove("a resolution row whose question is not in the question set", len(resolved))
    return {
        "question_set": str(questions.get("question_set")),
        "forecast_due_date": str(questions.get("forecast_due_date")),
        "dataset_questions": total,
        "removed": dict(sorted(removed.items())),
        "questions": sorted(kept, key=lambda d: (d["source"], d["id"])),
    }


def dumps(doc: dict[str, Any]) -> str:
    """Sorted keys; each question on a line of its own (byte-stable)."""
    lines = ["{"]
    keys = sorted(doc)
    for n, key in enumerate(keys):
        value = doc[key]
        end = "," if n < len(keys) - 1 else ""
        if isinstance(value, list):
            lines.append(f" {json.dumps(key)}: [")
            for m, item in enumerate(value):
                comma = "," if m < len(value) - 1 else ""
                lines.append("  " + json.dumps(item, sort_keys=True, ensure_ascii=False) + comma)
            lines.append(f" ]{end}")
        else:
            body = json.dumps(value, sort_keys=True, indent=1, ensure_ascii=False)
            lines.append(f" {json.dumps(key)}: " + body.replace("\n", "\n ") + end)
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fetched", required=True, help="the day the files were fetched")
    args = parser.parse_args(argv)
    sets = sorted(
        p.name.removesuffix("-llm.json")
        for p in args.raw.glob("2026-*-llm.json")
        if FIRST <= p.name[:10] <= LAST
    )
    if not sets:
        raise ExtractError(f"no question set from {FIRST} to {LAST} in {args.raw}")
    args.out.mkdir(parents=True, exist_ok=True)
    for s in sets:
        qpath, rpath = args.raw / f"{s}-llm.json", args.raw / f"{s}-res.json"
        if not rpath.exists():
            raise ExtractError(f"{s}: no resolution set {rpath.name}")
        doc = cut(
            json.loads(qpath.read_text(encoding="utf-8")),
            json.loads(rpath.read_text(encoding="utf-8")),
            args.fetched,
        )
        if doc["forecast_due_date"] != s:
            raise ExtractError(f"{qpath.name} is due {doc['forecast_due_date']}, not {s}")
        doc["role"] = "corpus" if s >= CORPUS_FIRST else "history"
        doc["provenance"] = {
            "licence": LICENCE,
            "fetched": args.fetched,
            "extracted_by": "scripts/extract_dataset_questions.py",
            "questions": {
                "url": f"{BASE}/question_sets/{s}-llm.json",
                "bytes": qpath.stat().st_size,
                "sha256": sha256(qpath),
            },
            "resolutions": {
                "url": f"{BASE}/resolution_sets/{s}_resolution_set.json",
                "bytes": rpath.stat().st_size,
                "sha256": sha256(rpath),
            },
        }
        (args.out / f"{s}.json").write_text(dumps(doc), encoding="utf-8")
        n = sum(len(q["resolutions"]) for q in doc["questions"])
        print(f"{s} ({doc['role']}): {len(doc['questions'])} questions, {n} instances")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
