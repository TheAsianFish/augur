"""Cut ForecastBench's processed forecast sets down to the questions Augur forecast.

    python -I scripts/extract_forecast_sets.py SETS_DIR TARBALL OUT_DIR \\
        --subsets data/forecastbench --last-modified "Tue, 17 Nov 2026 08:27:21 GMT" \\
        --fetched 2026-11-23

``SETS_DIR`` is the extracted ``forecastbench-processed-forecast-sets``
directory (one sub-directory per question set, one JSON file per entrant),
``TARBALL`` the archive it came from (hashed, never opened). For every set
with a committed ``<set>-llm.subset.json`` under ``--subsets``, writes
``OUT_DIR/<set>.json``: each entrant's organization, model, group and
leaderboard flag, and for each market question of the subset its outcome as
ForecastBench resolved it, the market value on the forecast due date, and
every entrant's forecast with its imputed flag. A provenance header names
the archive (URL, size, SHA-256, last-modified, fetch date) and its licence.

With ``--dataset DIR --dataset-out OUT`` (M15) it also cuts the dataset
instances: for every corpus set of M15's extract in ``DIR``
(``scripts/extract_dataset_questions.py``), ``OUT/<set>.json`` holds the
forecasts of ForecastBench's naive, imputed and always-0.5 baselines and of
the external entries named in :data:`DATASET_ENTRANTS` (where the set holds
them) on each (source, id, resolution date) instance, with the outcome as
the forecast set gives it and each forecast's imputed flag.

The files are untrusted data: read only as JSON, standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

URL = (
    "https://www.forecastbench.org/assets/data/processed-forecast-sets/"
    "processed_forecast_sets.tar.gz"
)
LICENCE = "CC BY-SA 4.0"
ATTRIBUTION = (
    "ForecastBench processed forecast sets, Forecasting Research Institute "
    "(forecastbench.org/datasets), CC BY-SA 4.0; cut down to the question instances "
    "Augur's backtests forecast, values unchanged"
)
MARKETS = frozenset({"manifold", "metaculus", "polymarket", "infer"})
BASELINES = {
    "always-0.5",
    "always-0",
    "always-1",
    "imputed-forecaster",
    "naive-forecaster",
    "random-uniform",
}
_FREEZE = re.compile(r"zero[-_]shot[-_]with[-_]freeze[-_]values\.json$")
_ZERO = re.compile(r"zero[-_]shot\.json$")
_SEARCH = re.compile(r"web-(x-)?search")

GROUPS = ("zero-shot", "zero-shot-freeze", "external", "baseline")


class ExtractError(ValueError):
    """The input is not what this script expects; nothing is written."""


def group_of(file: str) -> str:
    """The entrant group from the file name (``file`` without the set prefix)."""
    if file.startswith("external."):
        return "external"
    if file.startswith("ForecastBench.") and file[len("ForecastBench.") : -5] in BASELINES:
        return "baseline"
    if _FREEZE.search(file):
        return "zero-shot-freeze"
    if _ZERO.search(file):
        return "zero-shot"
    raise ExtractError(f"{file}: not a zero-shot, external or baseline file; classify it first")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_keys(subset: Path) -> list[tuple[str, str]]:
    """The market questions of a committed question subset, in its order."""
    data = json.loads(subset.read_text(encoding="utf-8"))
    return [
        (str(q["source"]), str(q["id"]))
        for q in data["questions"]
        if q.get("source") in MARKETS and isinstance(q.get("id"), str)
    ]


def _same(question: dict[str, Any], field: str, value: Any, file: str) -> None:
    if field not in question:
        question[field] = value
    elif question[field] != value:
        raise ExtractError(
            f"{file}: {question['source']}:{question['id']} has {field} {value!r}, "
            f"another file {question[field]!r}"
        )


def extract_set(
    set_dir: Path, question_set: str, keys: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    """One set's entrants and their forecasts on ``keys``."""
    wanted = set(keys)
    entrants: list[dict[str, Any]] = []
    rows: dict[tuple[str, str], dict[str, Any]] = {
        k: {"source": k[0], "id": k[1], "forecasts": [], "imputed": []} for k in keys
    }
    prefix = f"{question_set}."
    for path in sorted(set_dir.glob("*.json")):
        if not path.name.startswith(prefix):
            raise ExtractError(f"{path.name} is not a {question_set} file")
        file = path.name[len(prefix) :]
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("question_set") != f"{question_set}-llm.json":
            raise ExtractError(f"{path.name} is for {data.get('question_set')!r}")
        index = len(entrants)
        entrants.append(
            {
                "file": file,
                "organization": str(data["organization"]),
                "model": str(data["model"]),
                "model_organization": str(data.get("model_organization") or ""),
                "leaderboard_eligible": bool(data.get("leaderboard_eligible")),
                "group": group_of(file),
                "web_search": bool(_SEARCH.search(str(data["model"]))),
            }
        )
        got: dict[tuple[str, str], tuple[float, bool]] = {}
        for f in data["forecasts"]:
            key = (str(f.get("source")), str(f.get("id")))
            if key not in wanted:
                continue
            if key in got:
                raise ExtractError(f"{path.name}: {key[0]}:{key[1]} appears twice")
            if f.get("forecast") is None:
                continue
            got[key] = (float(f["forecast"]), bool(f.get("imputed")))
            q = rows[key]
            _same(q, "resolution_date", str(f.get("resolution_date"))[:10], path.name)
            _same(q, "resolved", bool(f.get("resolved")), path.name)
            _same(q, "resolved_to", f.get("resolved_to"), path.name)
            _same(q, "market_value_on_due_date", f.get("market_value_on_due_date"), path.name)
        for key in keys:
            value = got.get(key)
            rows[key]["forecasts"].append(None if value is None else value[0])
            if value is not None and value[1]:
                rows[key]["imputed"].append(index)
    missing = [f"{k[0]}:{k[1]}" for k in keys if "resolved_to" not in rows[k]]
    if missing:
        raise ExtractError(f"{question_set}: no entrant forecast {', '.join(missing[:3])}")
    return {"question_set": question_set, "entrants": entrants, "questions": list(rows.values())}


DATASET_BASELINES = ("naive-forecaster", "imputed-forecaster", "always-0.5")
DATASET_ENTRANTS = (
    ("Anonymous 24", "model-4"),
    ("NeuroFusion", "hybrid ensemble 1"),
    ("Anonymous 39", "stat-v1 (calibrated market anchor + dataset base rates)"),
    ("Anonymous 23", "undertow-mix (statbase x gpt-5.6-sol)"),
)
"""M15's comparators beside the baselines (PLAN.md, ``m15-registration``, item 8)."""


def dataset_keys(extract: Path) -> list[tuple[str, str, str]]:
    """The (source, id, resolution date) instances of one set of M15's extract."""
    data = json.loads(extract.read_text(encoding="utf-8"))
    return [
        (str(q["source"]), str(q["id"]), day)
        for q in data["questions"]
        for day in sorted(q["resolutions"])
    ]


def _wanted(file: str, data: dict[str, Any]) -> bool:
    if file.startswith("ForecastBench."):
        return file[len("ForecastBench.") : -5] in DATASET_BASELINES
    return (str(data.get("organization")), str(data.get("model"))) in DATASET_ENTRANTS


def extract_dataset_set(
    set_dir: Path, question_set: str, keys: Sequence[tuple[str, str, str]]
) -> dict[str, Any]:
    """One set's baseline and named-entrant forecasts on the dataset instances ``keys``."""
    wanted = set(keys)
    entrants: list[dict[str, Any]] = []
    rows: dict[tuple[str, str, str], dict[str, Any]] = {
        k: {"source": k[0], "id": k[1], "resolution_date": k[2], "forecasts": [], "imputed": []}
        for k in keys
    }
    prefix = f"{question_set}."
    for path in sorted(set_dir.glob("*.json")):
        if not path.name.startswith(prefix):
            raise ExtractError(f"{path.name} is not a {question_set} file")
        file = path.name[len(prefix) :]
        if not (file.startswith("ForecastBench.") or file.startswith("external.")):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not _wanted(file, data):
            continue
        if data.get("question_set") != f"{question_set}-llm.json":
            raise ExtractError(f"{path.name} is for {data.get('question_set')!r}")
        index = len(entrants)
        entrants.append(
            {
                "file": file,
                "organization": str(data["organization"]),
                "model": str(data["model"]),
                "leaderboard_eligible": bool(data.get("leaderboard_eligible")),
                "group": group_of(file),
            }
        )
        got: dict[tuple[str, str, str], tuple[float, bool]] = {}
        for f in data["forecasts"]:
            if not isinstance(f.get("id"), str) or f.get("direction"):
                continue
            key = (str(f.get("source")), str(f["id"]), str(f.get("resolution_date"))[:10])
            if key not in wanted or f.get("forecast") is None:
                continue
            if key in got:
                raise ExtractError(f"{path.name}: {key} appears twice")
            got[key] = (float(f["forecast"]), bool(f.get("imputed")))
            row = rows[key]
            _same(row, "resolved_to", f.get("resolved_to"), path.name)
        for key in keys:
            value = got.get(key)
            rows[key]["forecasts"].append(None if value is None else value[0])
            if value is not None and value[1]:
                rows[key]["imputed"].append(index)
    return {"question_set": question_set, "entrants": entrants, "instances": list(rows.values())}


def dumps(doc: dict[str, Any]) -> str:
    """Sorted keys; each entrant and each question on a line of its own (byte-stable)."""
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


def provenance(tarball: Path, last_modified: str, fetched: str) -> dict[str, Any]:
    return {
        "url": URL,
        "bytes": tarball.stat().st_size,
        "sha256": sha256(tarball),
        "last_modified": last_modified,
        "fetched": fetched,
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "extracted_by": "scripts/extract_forecast_sets.py",
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sets_dir", type=Path)
    parser.add_argument("tarball", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--subsets", type=Path, default=Path("data/forecastbench"))
    parser.add_argument("--last-modified", required=True)
    parser.add_argument("--fetched", required=True)
    parser.add_argument("--dataset", type=Path, help="M15's dataset extract directory")
    parser.add_argument("--dataset-out", type=Path, help="where the dataset cut goes")
    args = parser.parse_args(argv)
    if (args.dataset is None) != (args.dataset_out is None):
        raise ExtractError("--dataset and --dataset-out go together")
    head = provenance(args.tarball, args.last_modified, args.fetched)
    args.out.mkdir(parents=True, exist_ok=True)
    for subset in sorted(args.subsets.glob("*-llm.subset.json")):
        question_set = subset.name.removesuffix("-llm.subset.json")
        set_dir = args.sets_dir / question_set
        if not set_dir.is_dir():
            print(f"{question_set}: no forecast set in {args.sets_dir}; skipped")
            continue
        doc = extract_set(set_dir, question_set, subset_keys(subset))
        doc["provenance"] = head
        (args.out / f"{question_set}.json").write_text(dumps(doc), encoding="utf-8")
        print(f"{question_set}: {len(doc['entrants'])} entrants, {len(doc['questions'])} questions")
    if args.dataset is not None:
        args.dataset_out.mkdir(parents=True, exist_ok=True)
        for extract in sorted(args.dataset.glob("2026-*.json")):
            if json.loads(extract.read_text(encoding="utf-8")).get("role") != "corpus":
                continue
            question_set = extract.stem
            set_dir = args.sets_dir / question_set
            if not set_dir.is_dir():
                raise ExtractError(f"{question_set}: no forecast set in {args.sets_dir}")
            doc = extract_dataset_set(set_dir, question_set, dataset_keys(extract))
            doc["provenance"] = head
            (args.dataset_out / f"{question_set}.json").write_text(dumps(doc), encoding="utf-8")
            names = ", ".join(e["model"] for e in doc["entrants"])
            print(f"{question_set} (dataset): {len(doc['instances'])} instances; {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
