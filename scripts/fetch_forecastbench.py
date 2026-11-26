"""Download one ForecastBench question set and its resolution set, keep the eligible questions.

    python scripts/fetch_forecastbench.py 2026-05-24 data/forecastbench [ENDPOINT:MODEL]
    python scripts/fetch_forecastbench.py --open 2026-11-22 data/forecastbench ENDPOINT:MODEL

Writes ``<date>-llm.subset.json`` and ``<date>_resolution_set.subset.json``
(only the questions :func:`augur.forecastbench.load_set` keeps, every field
as published) and ``<date>.source.json`` with the URLs, the SHA-256 of each
full file and the filter counts. The data is ForecastBench's, CC BY-SA 4.0.

The release gate uses the model the set will be forecast with (default
``ollama:qwen2.5:7b``): its release date, or for a hosted model its OpenRouter
listing date, and ``model_release`` records which.

``--open`` (M16) fetches the question set alone, for a set forecast before
its outcomes are known: it never downloads the resolution set, even if one is
published, and writes ``<date>-llm.open.json`` (the questions
:func:`augur.forecastbench.load_open_set` keeps) and ``<date>.open.source.json``.

``--open-resolutions`` (M16, run only after both routes' forecasts on the set
are committed) fetches the resolution set of a set fetched with ``--open``,
keeps the rows of its questions and writes ``<date>_resolution_set.open.json``
and ``<date>.open.resolutions.json`` (when, the SHA-256 of the full file, and
the time of the upstream commit that last changed it).
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

from augur.forecastbench import LICENCE, load_open_set, load_set, open_subset, subset_files
from augur.llm.catalog import load_snapshot, release_bound
from augur.llm.endpoints import endpoints_from_env, parse_route

BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
MODEL = "ollama:qwen2.5:7b"
COMMITS = "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/commits"


def _get(url: str) -> bytes:
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        url, headers={"User-Agent": "augur-forecast (research)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        body: bytes = response.read()
    return body


def fetch_open(due: str, out: Path, released: date, release_note: str) -> int:
    """The question set only; the resolution set is not requested."""
    url = f"{BASE}/question_sets/{due}-llm.json"
    body = _get(url)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "questions.json"
        path.write_bytes(body)
        report = load_open_set(path, released=released)
        qs = open_subset(report, path)
    dump = {"indent": 1, "sort_keys": True, "ensure_ascii": False}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{due}-llm.open.json").write_text(json.dumps(qs, **dump) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    meta = {
        "licence": LICENCE,
        "urls": {"questions": url},
        "sha256": {"questions": hashlib.sha256(body).hexdigest()},
        "resolutions": "not requested: forecast before its outcomes are read",
        "model_release": release_note,
        "total": report.total,
        "removed": report.removed,
        "kept": len(report.kept),
    }
    (out / f"{due}.open.source.json").write_text(
        json.dumps(meta, **dump) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )
    print("\n".join(report.lines()))
    return 0


def fetch_open_resolutions(due: str, out: Path) -> int:
    """The resolution rows of the questions ``--open`` kept, with their provenance."""
    kept = json.loads((out / f"{due}-llm.open.json").read_text(encoding="utf-8"))
    keys = {(str(q.get("source")), q.get("id")) for q in kept["questions"]}
    path = f"datasets/resolution_sets/{due}_resolution_set.json"
    body = _get(f"{BASE.removesuffix('/datasets')}/{path}")
    commits = json.loads(
        _get(f"{COMMITS}?path={path}&per_page=1")  # the upstream commit that last changed it
    )
    rs = json.loads(body)
    rs["resolutions"] = [
        r for r in rs.get("resolutions", []) if (str(r.get("source")), r.get("id")) in keys
    ]
    dump = {"indent": 1, "sort_keys": True, "ensure_ascii": False}
    (out / f"{due}_resolution_set.open.json").write_text(
        json.dumps(rs, **dump) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )
    meta = {
        "licence": LICENCE,
        "fetched": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(body).hexdigest(),
        "upstream": {
            "url": f"{BASE}/resolution_sets/{due}_resolution_set.json",
            "commit": commits[0]["sha"] if commits else None,
            "committed": commits[0]["commit"]["committer"]["date"] if commits else None,
        },
        "rows": len(rs["resolutions"]),
        "resolved_0_or_1": sum(
            1 for r in rs["resolutions"] if r.get("resolved") and r.get("resolved_to") in (0, 1)
        ),
    }
    (out / f"{due}.open.resolutions.json").write_text(
        json.dumps(meta, **dump) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )
    print(f"{due}: {meta['rows']} rows for the open questions, {meta['resolved_0_or_1']} resolved")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "--open-resolutions":
        return fetch_open_resolutions(argv[1], Path(argv[2]))
    opened = bool(argv) and argv[0] == "--open"
    if opened:
        argv = argv[1:]
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    due, out = argv[0], Path(argv[1])
    date.fromisoformat(due)
    route = parse_route(argv[2] if len(argv) == 3 else MODEL, endpoints_from_env({}))
    released, basis = release_bound(route, load_snapshot())
    if released is None:
        raise SystemExit(f"no release date for {route.key}")
    note = f"{route.model} {released}" + ("" if basis == "release announcement" else f" ({basis})")
    if opened:
        return fetch_open(due, out, released, note)
    urls = {
        "questions": f"{BASE}/question_sets/{due}-llm.json",
        "resolutions": f"{BASE}/resolution_sets/{due}_resolution_set.json",
    }
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        digests = {}
        for kind, url in urls.items():
            body = _get(url)
            digests[kind] = hashlib.sha256(body).hexdigest()
            paths[kind] = Path(tmp) / f"{kind}.json"
            paths[kind].write_bytes(body)
        report = load_set(paths["questions"], paths["resolutions"], released=released)
        qs, rs = subset_files(report, paths["questions"], paths["resolutions"])
    dump = {"indent": 1, "sort_keys": True, "ensure_ascii": False}
    (out / f"{due}-llm.subset.json").write_text(json.dumps(qs, **dump) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    (out / f"{due}_resolution_set.subset.json").write_text(
        json.dumps(rs, **dump) + "\n",  # type: ignore[arg-type]
        encoding="utf-8",
    )
    meta = {
        "licence": LICENCE,
        "urls": urls,
        "sha256": digests,
        "model_release": note,
        "total": report.total,
        "removed": report.removed,
        "kept": len(report.kept),
    }
    (out / f"{due}.source.json").write_text(json.dumps(meta, **dump) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    print("\n".join(report.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
