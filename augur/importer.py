"""Rebuild one SQLite ledger from JSONL exports (the ``ledger`` branch).

Every scheduled run pushes its own export, so the live record is many small
files. Importing them in name order (each name starts with its UTC
timestamp) into one fresh ledger gives back a single file that ``augur
score`` and ``augur report`` read, with the leakage triggers re-checking
every source use and every citation on the way in.

Rows keep their ids, so importing the same export twice adds nothing: a row
whose primary key is already present is counted as a duplicate, and any
other constraint failure (a CHECK, a NOT NULL, a leakage trigger) aborts the
whole import. The one table whose integer id SQLite assigns
(``source_uses``) is matched on every other column instead, since two
exports number their rows independently.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from augur.clock import Clock
from augur.errors import LedgerError
from augur.export import read_jsonl
from augur.ledger import Ledger

# Tables whose integer primary key SQLite assigns; the id is dropped on import.
_AUTO_IDS: dict[str, str] = {"source_uses": "use_id"}
# Written by the migrations themselves, never imported.
_SKIP: frozenset[str] = frozenset({"schema_migrations"})


@dataclass
class ImportReport:
    files: int = 0
    inserted: dict[str, int] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.inserted.values())


def export_files(paths: Iterable[Path]) -> list[Path]:
    """Every ``*.jsonl`` under the given files and directories, in name order."""
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.jsonl"), key=lambda p: (p.name, str(p))))
        elif path.is_file():
            found.append(path)
        else:
            raise LedgerError(f"no export at {path}; pass JSONL files or a directory of them")
    return found


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(r[1]) for r in conn.execute(f'PRAGMA table_info("{table}")')]


def import_exports(paths: Iterable[Path], out: Path, clock: Clock) -> ImportReport:
    """Append every row of the exports to the ledger at ``out`` (created if absent)."""
    files = export_files(paths)
    report = ImportReport(files=len(files))
    with Ledger.open(out, clock) as ledger:
        conn = ledger.connection
        known = {
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        columns = {t: _columns(conn, t) for t in known}
        conn.execute("BEGIN")
        try:
            for path in files:
                for table, row in read_jsonl(path):
                    if table in _SKIP:
                        continue
                    if table not in known:
                        raise LedgerError(
                            f"{path}: table {table!r} is not in this ledger's schema; "
                            "upgrade Augur before importing a newer export"
                        )
                    _insert(conn, table, row, columns[table], report)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return report


def _insert(
    conn: sqlite3.Connection,
    table: str,
    row: dict[str, Any],
    columns: list[str],
    report: ImportReport,
) -> None:
    unknown = set(row) - set(columns)
    if unknown:
        raise LedgerError(
            f"{table} row has columns this schema lacks ({', '.join(sorted(unknown))}); "
            "upgrade Augur before importing"
        )
    auto = _AUTO_IDS.get(table)
    values = {k: v for k, v in row.items() if k != auto}
    names = list(values)
    if auto is not None:
        # Same content already present: a re-import, not a new row. "IS" treats
        # NULLs as equal.
        match = {n: values[n] for n in names}
    else:
        keys = _primary_key(conn, table)
        missing = [k for k in keys if k not in values]
        if missing:
            raise LedgerError(f"{table} row lacks its key column(s) {', '.join(missing)}")
        match = {k: values[k] for k in keys}
    where = " AND ".join(f'"{n}" IS ?' for n in match)
    hit = conn.execute(
        f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1',  # noqa: S608 - names from the schema
        tuple(match.values()),
    ).fetchone()
    if hit:
        report.duplicates[table] = report.duplicates.get(table, 0) + 1
        return
    quoted = ", ".join(f'"{n}"' for n in names)
    marks = ", ".join("?" for _ in names)
    sql = f'INSERT INTO "{table}" ({quoted}) VALUES ({marks})'  # noqa: S608
    try:
        cur = conn.execute(sql, tuple(values.values()))
    except sqlite3.IntegrityError as exc:
        raise LedgerError(f"import refused a {table} row: {exc}") from exc
    if cur.rowcount:
        report.inserted[table] = report.inserted.get(table, 0) + 1


def _primary_key(conn: sqlite3.Connection, table: str) -> list[str]:
    info = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    keys = [str(c[1]) for c in sorted(info, key=lambda c: int(c[5])) if int(c[5]) > 0]
    if not keys:
        raise LedgerError(f"table {table} has no primary key to match imported rows on")
    return keys
