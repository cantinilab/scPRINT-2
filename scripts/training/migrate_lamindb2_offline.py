#!/usr/bin/env python3
"""Offline, create-only LaminDB v1 -> v2 SQLite migration receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import UUID

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


def snapshot(path: Path) -> dict:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table' order by name"
            )
        ]
        counts = {
            name: connection.execute(f"select count(*) from {name}").fetchone()[0]
            for name in PRESERVED_TABLES
        }
        integrity = connection.execute("pragma integrity_check").fetchone()[0]
        migrations = connection.execute(
            "select count(*) from django_migrations"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "tables": tables,
        "counts": counts,
        "integrity": integrity,
        "migrations": migrations,
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def create_candidate(original: Path, candidate: Path) -> None:
    if candidate.exists():
        raise FileExistsError(candidate)
    temporary = candidate.with_name(f".{candidate.name}.{os.getpid()}.copy")
    if temporary.exists():
        raise FileExistsError(temporary)
    subprocess.run(
        [
            "cp",
            "--reflink=auto",
            "--preserve=mode,timestamps",
            str(original),
            str(temporary),
        ],
        check=True,
    )
    try:
        os.link(temporary, candidate)
        candidate_stat = candidate.stat()
        temporary_stat = temporary.stat()
        if (candidate_stat.st_dev, candidate_stat.st_ino) != (
            temporary_stat.st_dev,
            temporary_stat.st_ino,
        ):
            raise RuntimeError("candidate adoption did not preserve the copied inode")
    finally:
        temporary.unlink(missing_ok=True)


def deploy_offline(candidate: Path) -> None:
    # Construct a local-only instance explicitly. This avoids importing lamindb,
    # loading user credentials, refreshing a hub token, or syncing cloud SQLite.
    from django.core.management import call_command
    from django.db import connections
    from lamindb_setup._check_setup import disable_auto_connect
    from lamindb_setup.core._settings import settings
    from lamindb_setup.core._settings_instance import InstanceSettings
    from lamindb_setup.core.django import setup_django

    os.environ["LAMINDB_DJANGO_DATABASE_URL"] = f"sqlite:///{candidate.resolve()}"
    instance = InstanceSettings(
        id=UUID("00000000-0000-0000-0000-000000000000"),
        owner="none",
        name="none",
        storage=None,
        db=None,
        modules="bionty",
        is_on_hub=False,
        api_url=None,
    )
    settings._instance_settings = instance
    disable_auto_connect(setup_django)(instance, configure_only=True)
    call_command("migrate", verbosity=2, interactive=False)
    connections.close_all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("expected_original_sha256")
    args = parser.parse_args()

    original = args.original.resolve()
    candidate = args.candidate.resolve()
    receipt = args.receipt.resolve()
    if not original.is_file() or original.stat().st_size == 0:
        raise FileNotFoundError(original)
    if candidate.exists():
        raise FileExistsError(candidate)
    if receipt.exists():
        raise FileExistsError(receipt)

    original_stat = original.stat()
    original_sha = sha256(original)
    if original_sha != args.expected_original_sha256:
        raise RuntimeError(
            f"original SHA-256 mismatch: {original_sha} != {args.expected_original_sha256}"
        )
    before = snapshot(original)
    if before["integrity"] != "ok":
        raise RuntimeError(f"original integrity check failed: {before['integrity']}")
    if "lamindb_branch" in before["tables"]:
        raise RuntimeError("original unexpectedly already contains lamindb_branch")

    create_candidate(original, candidate)
    copied_sha = sha256(candidate)
    if copied_sha != original_sha:
        raise RuntimeError(f"candidate copy SHA-256 mismatch: {copied_sha} != {original_sha}")

    deploy_offline(candidate)
    after = snapshot(candidate)
    if after["integrity"] != "ok":
        raise RuntimeError(f"candidate integrity check failed: {after['integrity']}")
    if before["counts"] != after["counts"]:
        raise RuntimeError(
            f"preserved row counts changed: before={before['counts']} after={after['counts']}"
        )
    if "lamindb_branch" not in after["tables"]:
        raise RuntimeError("migration did not add lamindb_branch")
    if len(after["tables"]) <= len(before["tables"]):
        raise RuntimeError("migration did not increase the table count")
    if after["migrations"] <= before["migrations"]:
        raise RuntimeError("migration did not advance django_migrations")

    final_original_stat = original.stat()
    final_original_sha = sha256(original)
    if final_original_sha != original_sha:
        raise RuntimeError("original SHA-256 changed during migration")
    if (
        final_original_stat.st_size != original_stat.st_size
        or final_original_stat.st_mtime_ns != original_stat.st_mtime_ns
    ):
        raise RuntimeError("original size or mtime changed during migration")

    payload = {
        "status": "accepted_candidate",
        "mode": "offline-local-instance-direct-django-migrate",
        "network_required": False,
        "swapped": False,
        "original": str(original),
        "candidate": str(candidate),
        "original_sha256": original_sha,
        "candidate_sha256": sha256(candidate),
        "original_size": original_stat.st_size,
        "original_mtime_ns": original_stat.st_mtime_ns,
        "before_table_count": len(before["tables"]),
        "after_table_count": len(after["tables"]),
        "before_migrations": before["migrations"],
        "after_migrations": after["migrations"],
        "row_counts": after["counts"],
        "integrity": after["integrity"],
        "branch_table_added": True,
        "python": sys.version,
        "helper_sha256": sha256(Path(__file__).resolve()),
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt, payload)
    print("LAMINDB2_OFFLINE_MIGRATION_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
