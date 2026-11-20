"""Draw the label sample for the leakage judge, before reading any judge score.

For each arm, every source its forecasters were shown on the fixed sample
is ordered by the SHA-256 of its source id (unrelated to content or score)
and the first N are taken. The script writes a labelling sheet (question,
open instant, resolution, the source as the forecasters saw it) and a JSON
file of labels with ``score`` left null, to be filled by a rater blind to the judge on El Lahib
et al.'s 0 to 4 rubric (augur/judge.py). It never reads ``leak_judgments``.

    uv run python scripts/judge_sample.py --per-arm 20 \
        --sheet /tmp/sheet.md --out data/judge/m6-reference-labels.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from augur.backtest import sample
from augur.forecastbench import load_set
from augur.judge import load_source, shown_sources
from augur.report import Q2_ARMS, Q2_SAMPLE, Q2_SET


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-arm", type=int, default=20)
    ap.add_argument("--arms", default="O,T")
    ap.add_argument("--sheet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = load_set(
        Path(f"{Q2_SET}-llm.subset.json"),
        Path(f"{Q2_SET}_resolution_set.subset.json"),
        released=date.min,
    )
    questions = {q.question_id: q for q in sample(report.kept, Q2_SAMPLE)}
    labels: list[dict[str, object]] = []
    sheet: list[str] = []
    for arm in args.arms.split(","):
        path, tournament = Q2_ARMS[arm]
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
            pool = [
                (s.question_id, sid)
                for s in shown_sources(conn, tournament)
                if s.question_id in questions
                for sid in s.source_ids
            ]
            pool.sort(key=lambda x: hashlib.sha256(x[1].encode()).hexdigest())
            for qid, sid in pool[: args.per_arm]:
                fb = questions[qid]
                src = load_source(conn, sid)
                labels.append(
                    {"arm": arm, "question_id": qid, "source_id": sid, "score": None, "note": ""}
                )
                sheet += [
                    f"## {arm} {qid} {sid}",
                    f"Question: {fb.question}",
                    f"Opened (information cutoff): {fb.freeze_datetime:%Y-%m-%d}; resolved "
                    f"{'Yes' if fb.outcome else 'No'} on {fb.resolution_date}",
                    f"Source: {src.title} ({src.searcher}, dated "
                    f"{src.published_at:%Y-%m-%d} , {src.url})"
                    if src.published_at
                    else "",
                    "",
                    src.excerpt,
                    "",
                ]
    args.sheet.write_text("\n".join(sheet), encoding="utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rubric": "El Lahib et al. 0-4 (augur/judge.py); rated from the source text, blind "
        "to the judge's scores",
        "sample": f"first {args.per_arm} shown sources per arm by SHA-256 of the source id",
        "labels": labels,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"{len(labels)} sources to label; sheet {args.sheet}, labels {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
