"""Download one ForecastBench question set and its resolution set, keep the eligible questions.

    python scripts/fetch_forecastbench.py 2026-05-24 data/forecastbench

Writes ``<date>-llm.subset.json`` and ``<date>_resolution_set.subset.json``
(only the questions :func:`augur.forecastbench.load_set` keeps, every field
as published) and ``<date>.source.json`` with the URLs, the SHA-256 of each
full file and the filter counts. The data is ForecastBench's, CC BY-SA 4.0.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

from augur.forecastbench import LICENCE, load_set, subset_files
from augur.llm.registry import local_release_date

BASE = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
MODEL = "qwen2.5:7b"


def _get(url: str) -> bytes:
    request = urllib.request.Request(  # noqa: S310 - fixed https host
        url, headers={"User-Agent": "augur-forecast (research)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https host
        body: bytes = response.read()
    return body


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    due, out = argv[0], Path(argv[1])
    date.fromisoformat(due)
    released = local_release_date(MODEL)
    if released is None:
        raise SystemExit(f"no release date for {MODEL}")
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
        "model_release": f"{MODEL} {released}",
        "total": report.total,
        "removed": report.removed,
        "kept": len(report.kept),
    }
    (out / f"{due}.source.json").write_text(json.dumps(meta, **dump) + "\n", encoding="utf-8")  # type: ignore[arg-type]
    print("\n".join(report.lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
