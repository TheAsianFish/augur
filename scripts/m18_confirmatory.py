"""List M18's confirmatory instances from ForecastBench's question sets (no outcomes).

    python -I scripts/m18_confirmatory.py RAW_DIR OUT --fetched 2026-11-27

``RAW_DIR`` holds question sets as published, named ``<set>-llm.json``
(``forecastingresearch/forecastbench-datasets``, ``datasets/question_sets``).
A question set lists each question's resolution dates and holds no outcome,
so this script reads nothing that resolves anything. It writes ``OUT``: every
Meteo-France instance (``dbnomics``, an id starting ``meteofrance_TEMPERATURE``)
of a 2026 set whose resolution date falls from :data:`FIRST_DAY` to
:data:`LAST_DAY`, split into those on a question M15's committed extract holds
(the confirmatory set) and the rest (counted and left out), with each set
file's SHA-256 (PLAN.md, ``m18-registration``, item 7).

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

FIRST_DAY = "2026-11-28"
LAST_DAY = "2027-01-11"
PREFIX = "meteofrance_TEMPERATURE_"
BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"


def stratum(due: str, resolution: str) -> int:
    """The horizon in days, rounded to the registered strata (90, 180 or 365)."""
    from datetime import date

    days = (date.fromisoformat(resolution) - date.fromisoformat(due)).days
    for h in (90, 180, 365):
        if abs(days - h) <= 5:
            return h
    raise ValueError(f"{due} to {resolution} is {days} days, not a registered stratum")


def build(raw: Path, extract: Path, fetched: str) -> dict[str, Any]:
    held: set[tuple[str, str]] = set()
    for p in sorted(extract.glob("2026-*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        held.update((p.stem, str(q["id"])) for q in doc["questions"])
    instances: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    sets: dict[str, dict[str, Any]] = {}
    for p in sorted(raw.glob("2026-*-llm.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        s = str(doc["forecast_due_date"])[:10]
        found = 0
        for q in doc.get("questions", []):
            qid = q.get("id")
            if q.get("source") != "dbnomics" or not isinstance(qid, str):
                continue
            if not qid.startswith(PREFIX):
                continue
            for r in q.get("resolution_dates") or []:
                day = str(r)[:10]
                if not FIRST_DAY <= day <= LAST_DAY:
                    continue
                found += 1
                row = {
                    "question_set": s,
                    "id": qid,
                    "resolution_date": day,
                    "horizon": stratum(s, day),
                }
                (instances if (s, qid) in held else left_out).append(row)
        if found:
            sets[s] = {
                "url": f"{BASE}/question_sets/{s}-llm.json",
                "bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
    key = lambda d: (d["question_set"], d["id"], d["resolution_date"])  # noqa: E731
    return {
        "window": [FIRST_DAY, LAST_DAY],
        "fetched": fetched,
        "rule": "Meteo-France instances of the 2026 question sets resolving in the window, "
        "on a question M15's committed extract holds; the rest counted and left out",
        "question_sets": sets,
        "instances": sorted(instances, key=key),
        "left_out": sorted(left_out, key=key),
    }


GATE_SET = "2026-09-27"
SOURCES = frozenset({"acled", "dbnomics", "fred", "wikipedia", "yfinance"})


def gate(raw: Path, extract: Path) -> dict[str, Any]:
    """P65's instances: the 2026-09-27 set's single-series ones resolving in the window."""
    path = extract / f"{GATE_SET}.json"
    held = {
        (str(q["source"]), str(q["id"]))
        for q in json.loads(path.read_text(encoding="utf-8"))["questions"]
    }
    qpath = raw / f"{GATE_SET}-llm.json"
    doc = json.loads(qpath.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    left: list[dict[str, Any]] = []
    for q in doc.get("questions", []):
        src, qid = q.get("source"), q.get("id")
        if src not in SOURCES or not isinstance(qid, str):
            continue
        for r in q.get("resolution_dates") or []:
            day = str(r)[:10]
            if FIRST_DAY <= day <= LAST_DAY:
                row = {"source": src, "id": qid, "resolution_date": day}
                (rows if (src, qid) in held else left).append(row)
    key = lambda d: (d["source"], d["id"], d["resolution_date"])  # noqa: E731
    return {
        "question_set": GATE_SET,
        "window": [FIRST_DAY, LAST_DAY],
        "rule": "P65: the 2026-09-27 set's single-series instances of M15's sources resolving "
        "in the window, on a question M15's committed extract holds; the rest left out",
        "question_set_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
        "instances": sorted(rows, key=key),
        "left_out": sorted(left, key=key),
    }


def dumps(doc: dict[str, Any]) -> str:
    lines = ["{"]
    keys = sorted(doc)
    for n, k in enumerate(keys):
        end = "," if n < len(keys) - 1 else ""
        v = doc[k]
        if isinstance(v, list) and v and isinstance(v[0], dict):
            lines.append(f" {json.dumps(k)}: [")
            for m, item in enumerate(v):
                comma = "," if m < len(v) - 1 else ""
                lines.append("  " + json.dumps(item, sort_keys=True) + comma)
            lines.append(f" ]{end}")
        else:
            body = json.dumps(v, sort_keys=True, indent=1).replace("\n", "\n ")
            lines.append(f" {json.dumps(k)}: {body}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fetched", required=True)
    parser.add_argument("--extract", type=Path, default=Path("data/forecastbench/dataset"))
    parser.add_argument("--gate", type=Path, help="also write P65's instance list here")
    args = parser.parse_args(argv)
    if args.gate is not None:
        g = gate(args.raw, args.extract)
        args.gate.write_text(dumps(g), encoding="utf-8")
        print(f"P65: {len(g['instances'])} instances, {len(g['left_out'])} left out")
    doc = build(args.raw, args.extract, args.fetched)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(dumps(doc), encoding="utf-8")
    print(f"{len(doc['instances'])} confirmatory instances, {len(doc['left_out'])} left out")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
