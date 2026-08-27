#!/usr/bin/env python3
"""Fail closed unless a LaminDB SQLite matches its accepted migration receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

PRESERVED_TABLES = (
    "lamindb_artifact",
    "lamindb_collection",
    "lamindb_storage",
    "lamindb_transform",
    "lamindb_user",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(receipt_path: Path, database: Path) -> dict:
    receipt_path = receipt_path.resolve()
    database = database.resolve()
    if not receipt_path.is_file() or not database.is_file():
        raise FileNotFoundError(receipt_path if not receipt_path.is_file() else database)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "accepted_candidate":
        raise RuntimeError("migration receipt is not accepted_candidate")
    if receipt.get("swapped") is not False:
        raise RuntimeError("migration receipt must describe an unswapped candidate")
    if Path(receipt.get("candidate", "")).resolve() != database:
        raise RuntimeError("migration receipt candidate path does not match database")
    observed_sha = sha256(database)
    if receipt.get("candidate_sha256") != observed_sha:
        raise RuntimeError("database SHA-256 does not match migration receipt")

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=30)
    try:
        integrity = connection.execute("pragma quick_check").fetchone()[0]
        branch = connection.execute(
            "select 1 from sqlite_master where type='table' and name='lamindb_branch'"
        ).fetchone()
        migrations = connection.execute(
            "select count(*) from django_migrations"
        ).fetchone()[0]
        counts = {
            name: connection.execute(f"select count(*) from {name}").fetchone()[0]
            for name in PRESERVED_TABLES
        }
    finally:
        connection.close()
    if integrity != "ok":
        raise RuntimeError(f"database quick_check failed: {integrity}")
    if branch is None or receipt.get("branch_table_added") is not True:
        raise RuntimeError("lamindb_branch migration is not proven")
    if migrations != receipt.get("after_migrations"):
        raise RuntimeError("django_migrations count does not match receipt")
    if counts != receipt.get("row_counts"):
        raise RuntimeError("preserved table row counts do not match receipt")
    return {
        "status": "accepted",
        "database": str(database),
        "database_sha256": observed_sha,
        "migration_receipt": str(receipt_path),
        "migrations": migrations,
        "row_counts": counts,
        "integrity": integrity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    result = verify(args.receipt, args.database)
    print("MIGRATED_LAMINDB_GATE_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
