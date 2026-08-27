#!/usr/bin/env python3
"""Create and verify a fail-closed migrated LaminDB SQLite receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

DATA_PREFIXES = ("lamindb_", "bionty_")
EXPECTED_COLLECTION_KEY = "scPRINT-V2 (all+tahoe+scbase) filtered"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def database_snapshot(path: Path, *, full_integrity: bool) -> dict:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table' order by name"
            )
        ]
        data_tables = [
            name for name in tables if any(name.startswith(p) for p in DATA_PREFIXES)
        ]
        counts = {
            name: connection.execute(f"select count(*) from {name}").fetchone()[0]
            for name in data_tables
        }
        integrity_pragma = "integrity_check" if full_integrity else "quick_check"
        integrity = connection.execute(f"pragma {integrity_pragma}").fetchone()[0]
        migrations = connection.execute(
            "select count(*) from django_migrations"
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "all_tables": tables,
        "data_table_counts": counts,
        "data_table_counts_sha256": digest_json(counts),
        "integrity": integrity,
        "integrity_pragma": integrity_pragma,
        "migrations": migrations,
    }


def collection_membership(path: Path, key: str) -> dict:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        collections = connection.execute(
            "select id, uid, key from lamindb_collection where key = ? order by id", (key,)
        ).fetchall()
        if len(collections) != 1:
            raise RuntimeError(f"expected exactly one filtered collection, got {collections}")
        collection_id, collection_uid, collection_key = collections[0]
        rows = connection.execute(
            """
            select artifact.uid
            from lamindb_collectionartifact as link
            join lamindb_artifact as artifact on artifact.id = link.artifact_id
            where link.collection_id = ?
            order by artifact.uid
            """,
            (collection_id,),
        ).fetchall()
        link_count = connection.execute(
            "select count(*) from lamindb_collectionartifact where collection_id = ?",
            (collection_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    uids = [row[0] for row in rows]
    uid_payload = ("\n".join(uids) + "\n").encode()
    return {
        "collection_id": collection_id,
        "collection_uid": collection_uid,
        "collection_key": collection_key,
        "artifact_count": len(uids),
        "unique_artifact_count": len(set(uids)),
        "link_count": link_count,
        "artifact_uids_sha256": hashlib.sha256(uid_payload).hexdigest(),
        "artifact_uids": uids,
    }


def verify_duplicate_hash_canonicalization(
    original: Path, candidate: Path, audit: dict, protected_collection_key: str
) -> dict:
    expected_ranking = [
        "collection_link_count_desc",
        "is_latest_desc",
        "active_branch_desc",
        "id_asc",
    ]
    groups = audit.get("groups")
    if (
        audit.get("mode") != "candidate-only-exact-identity-shadow-hash-null"
        or audit.get("protected_collection_key") != protected_collection_key
        or audit.get("ranking") != expected_ranking
        or not isinstance(groups, list)
        or audit.get("duplicate_group_count") != len(groups)
        or audit.get("duplicate_row_count") != 2 * len(groups)
        or audit.get("canonicalized_shadow_count") != len(groups)
    ):
        raise RuntimeError("invalid duplicate hash canonicalization audit contract")

    original_connection = sqlite3.connect(f"file:{original}?mode=ro", uri=True, timeout=30)
    candidate_connection = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True, timeout=30)
    original_connection.row_factory = sqlite3.Row
    candidate_connection.row_factory = sqlite3.Row
    try:
        original_columns = {
            row[1] for row in original_connection.execute("pragma table_info(lamindb_artifact)")
        }
        candidate_columns = {
            row[1] for row in candidate_connection.execute("pragma table_info(lamindb_artifact)")
        }
        if "hash" not in original_columns or "hash" not in candidate_columns:
            if groups:
                raise RuntimeError("duplicate hash audit requires Artifact.hash")
            return {
                "duplicate_group_count": 0,
                "canonicalized_shadow_count": 0,
            }
        observed_hashes = [
            row["hash"]
            for row in original_connection.execute(
                """select hash from lamindb_artifact where hash is not null
                group by hash having count(*) > 1 order by hash"""
            )
        ]
        if observed_hashes != [group.get("hash") for group in groups]:
            raise RuntimeError("original duplicate hashes do not match canonicalization audit")
        recorded_ids = set()
        for group in groups:
            rows_before = group.get("rows_before")
            if not isinstance(rows_before, list) or len(rows_before) != 2:
                raise RuntimeError("canonicalization audit rows_before is invalid")
            original_rows = original_connection.execute(
                """select id, uid, hash, key, size, _hash_type from lamindb_artifact
                where hash = ? order by id""",
                (group["hash"],),
            ).fetchall()
            expected_rows = [
                {
                    key: row[key]
                    for key in ("id", "uid", "hash", "key", "size", "_hash_type")
                }
                for row in rows_before
            ]
            if [dict(row) for row in original_rows] != expected_rows:
                raise RuntimeError("original duplicate rows do not match canonicalization audit")
            keeper_id = group.get("keeper_id")
            shadow_id = group.get("shadow_id")
            if keeper_id == shadow_id or {keeper_id, shadow_id} != {
                row["id"] for row in original_rows
            }:
                raise RuntimeError("canonicalization keeper/shadow IDs are invalid")
            if recorded_ids.intersection({keeper_id, shadow_id}):
                raise RuntimeError("canonicalization audit repeats an artifact ID")
            recorded_ids.update({keeper_id, shadow_id})
            candidate_rows = candidate_connection.execute(
                """select id, uid, hash, key, size, _hash_type from lamindb_artifact
                where id in (?, ?) order by id""",
                (keeper_id, shadow_id),
            ).fetchall()
            if len(candidate_rows) != 2:
                raise RuntimeError("canonicalization candidate rows are missing")
            candidate_by_id = {row["id"]: row for row in candidate_rows}
            if candidate_by_id[keeper_id]["hash"] != group["hash"]:
                raise RuntimeError("canonicalization keeper hash mismatch")
            if candidate_by_id[shadow_id]["hash"] is not None:
                raise RuntimeError("canonicalization shadow hash mismatch")
            for original_row in original_rows:
                candidate_row = candidate_by_id[original_row["id"]]
                for field in ("uid", "key", "size", "_hash_type"):
                    if candidate_row[field] != original_row[field]:
                        raise RuntimeError(
                            f"canonicalization changed non-hash field {field}"
                        )
        remaining = candidate_connection.execute(
            """select hash from lamindb_artifact where hash is not null
            group by hash having count(*) > 1"""
        ).fetchall()
        if remaining:
            raise RuntimeError("candidate retains duplicate Artifact.hash values")
    finally:
        original_connection.close()
        candidate_connection.close()
    return {
        "duplicate_group_count": len(groups),
        "canonicalized_shadow_count": len(groups),
    }


def create_receipt(
    original: Path,
    candidate: Path,
    migration_receipt_path: Path,
    registry_manifest_path: Path,
    membership_contract_path: Path,
    output: Path,
) -> dict:
    paths = [
        original,
        candidate,
        migration_receipt_path,
        registry_manifest_path,
        membership_contract_path,
    ]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    if output.exists():
        raise FileExistsError(output)

    original = original.resolve()
    candidate = candidate.resolve()
    migration_receipt_path = migration_receipt_path.resolve()
    registry_manifest_path = registry_manifest_path.resolve()
    membership_contract_path = membership_contract_path.resolve()
    output = output.resolve()
    migration = json.loads(migration_receipt_path.read_text())
    registry = json.loads(registry_manifest_path.read_text())
    contract = json.loads(membership_contract_path.read_text())

    original_sha = sha256(original)
    candidate_sha = sha256(candidate)
    if migration.get("status") != "accepted_candidate":
        raise RuntimeError("migration receipt is not accepted_candidate")
    if migration.get("swapped") is not False:
        raise RuntimeError("migration receipt must describe an unswapped candidate")
    if Path(migration.get("original", "")).resolve() != original:
        raise RuntimeError("migration receipt original path mismatch")
    if Path(migration.get("candidate", "")).resolve() != candidate:
        raise RuntimeError("migration receipt candidate path mismatch")
    if migration.get("original_sha256") != original_sha:
        raise RuntimeError("original SHA-256 does not match migration receipt")
    if migration.get("candidate_sha256") != candidate_sha:
        raise RuntimeError("candidate SHA-256 does not match migration receipt")
    duplicate_hash_verification = verify_duplicate_hash_canonicalization(
        original,
        candidate,
        migration.get("duplicate_hash_canonicalization", {}),
        EXPECTED_COLLECTION_KEY,
    )

    if contract.get("collection") != EXPECTED_COLLECTION_KEY:
        raise RuntimeError("membership contract collection key mismatch")
    if registry.get("output_sha256") != original_sha:
        raise RuntimeError("registry manifest does not bind the original database")
    if Path(registry.get("output", "")).resolve() != original:
        raise RuntimeError("registry manifest original path mismatch")
    expected_count = contract.get("filtered_artifact_count")
    expected_digest = contract.get("filtered_artifact_uids_sha256")
    expected_cells = contract.get("filtered_cell_count")
    if registry.get("kept_count") != expected_count:
        raise RuntimeError("registry and membership artifact counts disagree")
    if registry.get("kept_uids_sha256") != expected_digest:
        raise RuntimeError("registry and membership UID digests disagree")
    if registry.get("inferred_cells") != expected_cells:
        raise RuntimeError("registry and membership cell counts disagree")

    before = database_snapshot(original, full_integrity=True)
    after = database_snapshot(candidate, full_integrity=True)
    if before["integrity"] != "ok" or after["integrity"] != "ok":
        raise RuntimeError("full SQLite integrity check failed")
    before_counts = before["data_table_counts"]
    after_counts = after["data_table_counts"]
    missing = sorted(set(before_counts) - set(after_counts))
    changed = {
        name: {"before": before_counts[name], "after": after_counts.get(name)}
        for name in before_counts
        if after_counts.get(name) != before_counts[name]
    }
    if missing or changed:
        raise RuntimeError(
            f"pre-existing LaminDB/Bionty table parity failed: missing={missing}, changed={changed}"
        )
    if "lamindb_branch" in before["all_tables"]:
        raise RuntimeError("original unexpectedly contains lamindb_branch")
    if "lamindb_branch" not in after["all_tables"]:
        raise RuntimeError("candidate lacks lamindb_branch")
    if after["migrations"] <= before["migrations"]:
        raise RuntimeError("candidate did not advance django_migrations")

    original_membership = collection_membership(original, EXPECTED_COLLECTION_KEY)
    candidate_membership = collection_membership(candidate, EXPECTED_COLLECTION_KEY)
    for observed in (original_membership, candidate_membership):
        if observed["collection_id"] != registry.get("filtered_collection_id"):
            raise RuntimeError("filtered collection ID does not match registry manifest")
        if observed["artifact_count"] != expected_count:
            raise RuntimeError("filtered collection artifact count mismatch")
        if observed["unique_artifact_count"] != expected_count:
            raise RuntimeError("filtered collection contains duplicate artifact UIDs")
        if observed["link_count"] != expected_count:
            raise RuntimeError("filtered collection link count mismatch")
        if observed["artifact_uids_sha256"] != expected_digest:
            raise RuntimeError("filtered collection UID digest mismatch")
    if original_membership != candidate_membership:
        raise RuntimeError("filtered collection membership changed during migration")
    excluded = {item["uid"] for item in contract.get("excluded_artifacts", [])}
    if excluded.intersection(candidate_membership["artifact_uids"]):
        raise RuntimeError("excluded artifact remains in filtered collection")

    public_membership = dict(candidate_membership)
    public_membership.pop("artifact_uids")
    payload = {
        "status": "accepted_migrated_lamindb",
        "candidate": str(candidate),
        "candidate_sha256": candidate_sha,
        "original": str(original),
        "original_sha256": original_sha,
        "migration_receipt": str(migration_receipt_path),
        "migration_receipt_sha256": sha256(migration_receipt_path),
        "registry_manifest": str(registry_manifest_path),
        "registry_manifest_sha256": sha256(registry_manifest_path),
        "membership_contract": str(membership_contract_path),
        "membership_contract_sha256": sha256(membership_contract_path),
        "filtered_cell_count": expected_cells,
        "membership": public_membership,
        "duplicate_hash_canonicalization": migration[
            "duplicate_hash_canonicalization"
        ],
        "duplicate_hash_verification": duplicate_hash_verification,
        "before_data_table_counts": before_counts,
        "before_data_table_counts_sha256": before["data_table_counts_sha256"],
        "candidate_data_table_counts": after_counts,
        "candidate_data_table_counts_sha256": after["data_table_counts_sha256"],
        "before_migrations": before["migrations"],
        "after_migrations": after["migrations"],
        "integrity": {"original": before["integrity"], "candidate": after["integrity"]},
        "framework_exclusions": {
            "django_migrations": "expected to advance during schema migration",
            "new_tables": "new migration tables may be added; every pre-existing lamindb_/bionty_ table is parity checked",
        },
        "swapped": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def verify(receipt_path: Path, database: Path) -> dict:
    receipt_path = receipt_path.resolve()
    database = database.resolve()
    if not receipt_path.is_file() or not database.is_file():
        raise FileNotFoundError(receipt_path if not receipt_path.is_file() else database)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "accepted_migrated_lamindb":
        raise RuntimeError("verification receipt is not accepted_migrated_lamindb")
    if receipt.get("swapped") is not False:
        raise RuntimeError("verification receipt must describe an unswapped candidate")
    if Path(receipt.get("candidate", "")).resolve() != database:
        raise RuntimeError("verification receipt candidate path does not match database")
    observed_sha = sha256(database)
    if receipt.get("candidate_sha256") != observed_sha:
        raise RuntimeError("database SHA-256 does not match verification receipt")
    duplicate_hash_verification = verify_duplicate_hash_canonicalization(
        Path(receipt.get("original", "")).resolve(),
        database,
        receipt.get("duplicate_hash_canonicalization", {}),
        EXPECTED_COLLECTION_KEY,
    )
    snapshot = database_snapshot(database, full_integrity=False)
    if snapshot["integrity"] != "ok":
        raise RuntimeError(f"database quick_check failed: {snapshot['integrity']}")
    if (
        snapshot["data_table_counts_sha256"]
        != receipt.get("candidate_data_table_counts_sha256")
    ):
        raise RuntimeError("database data-table counts do not match verification receipt")
    expected_membership = receipt.get("membership", {})
    membership = collection_membership(database, expected_membership.get("collection_key", ""))
    membership.pop("artifact_uids")
    if membership != expected_membership:
        raise RuntimeError("database collection membership does not match verification receipt")
    return {
        "status": "accepted",
        "database": str(database),
        "database_sha256": observed_sha,
        "verification_receipt": str(receipt_path),
        "data_table_counts_sha256": snapshot["data_table_counts_sha256"],
        "migrations": snapshot["migrations"],
        "membership": membership,
        "duplicate_hash_verification": duplicate_hash_verification,
        "integrity": snapshot["integrity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("original", type=Path)
    create.add_argument("candidate", type=Path)
    create.add_argument("migration_receipt", type=Path)
    create.add_argument("registry_manifest", type=Path)
    create.add_argument("membership_contract", type=Path)
    create.add_argument("output", type=Path)
    check = sub.add_parser("verify")
    check.add_argument("receipt", type=Path)
    check.add_argument("database", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create_receipt(
            args.original,
            args.candidate,
            args.migration_receipt,
            args.registry_manifest,
            args.membership_contract,
            args.output,
        )
        print("MIGRATED_LAMINDB_RECEIPT_PASS " + json.dumps(result, sort_keys=True))
    else:
        result = verify(args.receipt, args.database)
        print("MIGRATED_LAMINDB_GATE_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
