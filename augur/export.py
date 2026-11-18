"""Deterministic JSONL export of a ledger, and the reader the import uses.

One line per row, ``{"table": name, "row": {...}}``, tables in a fixed order
and rows in insertion (rowid) order, keys sorted, so the same ledger always
exports to the same bytes. Blobs are written as ``{"$base64": "..."}``.

    python -m augur.export LEDGER.sqlite OUT.jsonl
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from augur.ledger import APPEND_ONLY_TABLES

BLOB_MARKER = "$base64"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _tables(conn: sqlite3.Connection) -> list[str]:
    """Known tables in schema order, then any others by name; SQLite's own skipped."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    present = {str(r[0]) for r in rows}
    known = [t for t in APPEND_ONLY_TABLES if t in present]
    return known + sorted(present - set(known))


def _rows(conn: sqlite3.Connection, table: str) -> Iterator[dict[str, Any]]:
    # Table names come from sqlite_master, quoted; no input reaches the SQL.
    try:
        cur = conn.execute(f"SELECT * FROM {_quote(table)} ORDER BY rowid")  # noqa: S608
    except sqlite3.OperationalError:
        # A WITHOUT ROWID table: order by its primary key instead.
        info = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        pk = [str(c[1]) for c in sorted(info, key=lambda c: int(c[5])) if int(c[5]) > 0]
        order = ", ".join(_quote(c) for c in pk) or "1"
        cur = conn.execute(f"SELECT * FROM {_quote(table)} ORDER BY {order}")  # noqa: S608
    names = [str(d[0]) for d in cur.description]
    for values in cur:
        yield {name: _encode(v) for name, v in zip(names, values, strict=True)}


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {BLOB_MARKER: base64.b64encode(value).decode("ascii")}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {BLOB_MARKER}:
        return base64.b64decode(value[BLOB_MARKER])
    return value


def _line(table: str, row: dict[str, Any]) -> str:
    return json.dumps(
        {"table": table, "row": row}, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def export_jsonl(db_path: Path, out: Path) -> dict[str, int]:
    """Write every row of the ledger at ``db_path`` to ``out``; return rows per table."""
    if not db_path.is_file():
        raise FileNotFoundError(f"no ledger at {db_path}; pass the path of an Augur SQLite file")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    counts: dict[str, int] = {}
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for table in _tables(conn):
                counts[table] = 0
                for row in _rows(conn, table):
                    fh.write(_line(table, row) + "\n")
                    counts[table] += 1
        tmp.replace(out)
    finally:
        conn.close()
    return counts


def read_jsonl(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(table, row)`` from an export, blobs decoded back to bytes."""
    with path.open(encoding="utf-8") as fh:
        for number, text in enumerate(fh, start=1):
            if not text.strip():
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: not JSON ({exc.msg})") from exc
            if (
                not isinstance(obj, dict)
                or not isinstance(obj.get("table"), str)
                or not isinstance(obj.get("row"), dict)
            ):
                raise ValueError(f'{path}:{number}: expected {{"table": str, "row": {{...}}}}')
            yield obj["table"], {k: _decode(v) for k, v in obj["row"].items()}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: python -m augur.export LEDGER.sqlite OUT.jsonl", file=sys.stderr)
        return 2
    counts = export_jsonl(Path(args[0]), Path(args[1]))
    total = sum(counts.values())
    detail = ", ".join(f"{t} {n}" for t, n in counts.items() if n)
    print(f"exported {total} rows to {args[1]} ({detail or 'empty'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
