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

DATA_PREFIXES = ("lamindb_", "bionty_")
EXPECTED_COLLECTION_KEY = "scPRINT-V2 (all+tahoe+scbase) filtered"


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
        data_tables = [
            name for name in tables if any(name.startswith(p) for p in DATA_PREFIXES)
        ]
        counts = {
            name: connection.execute(f"select count(*) from {name}").fetchone()[0]
            for name in data_tables
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


def canonicalize_duplicate_artifact_hashes(
    candidate: Path, protected_collection_key: str
) -> dict:
    """Null only exact-identity shadow hashes in the new candidate.

    LaminDB 2 makes ``Artifact.hash`` unique. Legacy registries may contain two
    records for the same bytes, so rank each exact pair deterministically and
    retain the hash on one canonical record. No protected training-collection
    member is eligible for this compatibility repair.
    """
    connection = sqlite3.connect(candidate, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("begin immediate")
        collections = connection.execute(
            "select id from lamindb_collection where key = ? order by id",
            (protected_collection_key,),
        ).fetchall()
        if len(collections) != 1:
            raise RuntimeError(
                f"expected exactly one protected collection, got {len(collections)}"
            )
        protected_collection_id = collections[0]["id"]
        duplicate_hashes = [
            row["hash"]
            for row in connection.execute(
                """select hash from lamindb_artifact where hash is not null
                group by hash having count(*) > 1 order by hash"""
            )
        ]
        groups = []
        for artifact_hash in duplicate_hashes:
            rows = connection.execute(
                """select artifact.id, artifact.uid, artifact.hash, artifact.key,
                artifact.size, artifact._hash_type, artifact.is_latest,
                artifact._branch_code,
                (select count(*) from lamindb_collectionartifact as link
                 where link.artifact_id = artifact.id) as collection_link_count,
                (select count(*) from lamindb_collectionartifact as link
                 where link.artifact_id = artifact.id and link.collection_id = ?)
                 as protected_collection_link_count
                from lamindb_artifact as artifact
                where artifact.hash = ? order by artifact.id""",
                (protected_collection_id, artifact_hash),
            ).fetchall()
            if len(rows) != 2:
                raise RuntimeError(
                    f"duplicate hash group must be an exact pair: {artifact_hash}"
                )
            identities = {(row["key"], row["size"], row["_hash_type"]) for row in rows}
            if len(identities) != 1:
                raise RuntimeError(
                    f"duplicate hash group does not have exact content identity: {artifact_hash}"
                )
            if any(row["protected_collection_link_count"] for row in rows):
                raise RuntimeError(
                    f"duplicate hash group intersects protected collection: {artifact_hash}"
                )
            ranked = sorted(
                rows,
                key=lambda row: (
                    -row["collection_link_count"],
                    -int(row["is_latest"]),
                    -int(row["_branch_code"] == 1),
                    row["id"],
                ),
            )
            keeper, shadow = ranked
            before = [dict(row) for row in rows]
            cursor = connection.execute(
                "update lamindb_artifact set hash = null where id = ? and hash = ?",
                (shadow["id"], artifact_hash),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"failed to canonicalize shadow row {shadow['id']}")
            groups.append(
                {
                    "hash": artifact_hash,
                    "content_identity": {
                        "key": keeper["key"],
                        "size": keeper["size"],
                        "hash_type": keeper["_hash_type"],
                    },
                    "keeper_id": keeper["id"],
                    "keeper_uid": keeper["uid"],
                    "shadow_id": shadow["id"],
                    "shadow_uid": shadow["uid"],
                    "rows_before": before,
                    "shadow_hash_after": None,
                }
            )
        remaining = connection.execute(
            """select hash, count(*) from lamindb_artifact where hash is not null
            group by hash having count(*) > 1"""
        ).fetchall()
        if remaining:
            raise RuntimeError(f"duplicate hashes remain after canonicalization: {remaining}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "mode": "candidate-only-exact-identity-shadow-hash-null",
        "protected_collection_key": protected_collection_key,
        "protected_collection_id": protected_collection_id,
        "ranking": [
            "collection_link_count_desc",
            "is_latest_desc",
            "active_branch_desc",
            "id_asc",
        ],
        "duplicate_group_count": len(groups),
        "duplicate_row_count": sum(len(group["rows_before"]) for group in groups),
        "canonicalized_shadow_count": len(groups),
        "groups": groups,
    }


def canonicalize_duplicate_artifact_schema_slots(
    candidate: Path, protected_collection_key: str
) -> dict:
    """Null only strict-subset schema slots that block LaminDB's new uniqueness.

    LaminDB 2 requires one non-null ``ArtifactSchema.slot`` per artifact. Legacy
    registries can link both an older schema and a strict feature superset at the
    same slot. Preserve both links and schema identities, retain the superset as
    the authoritative slotted link, and make only the subset link slotless.
    """
    connection = sqlite3.connect(candidate, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("begin immediate")
        collections = connection.execute(
            "select id from lamindb_collection where key = ? order by id",
            (protected_collection_key,),
        ).fetchall()
        if len(collections) != 1:
            raise RuntimeError(
                f"expected exactly one protected collection, got {len(collections)}"
            )
        protected_collection_id = collections[0]["id"]
        duplicate_groups = connection.execute(
            """select artifact_id, slot, count(*) as row_count
            from lamindb_artifactschema where slot is not null
            group by artifact_id, slot having count(*) > 1
            order by artifact_id, slot"""
        ).fetchall()
        groups = []
        row_fields = (
            "id",
            "artifact_id",
            "schema_id",
            "slot",
            "feature_ref_is_semantic",
            "run_id",
            "created_by_id",
            "created_at",
        )
        for duplicate in duplicate_groups:
            artifact_id = duplicate["artifact_id"]
            slot = duplicate["slot"]
            if duplicate["row_count"] != 2:
                raise RuntimeError(
                    f"duplicate artifact-schema slot must be an exact pair: artifact={artifact_id} slot={slot}"
                )
            protected_links = connection.execute(
                """select count(*) from lamindb_collectionartifact
                where artifact_id = ? and collection_id = ?""",
                (artifact_id, protected_collection_id),
            ).fetchone()[0]
            if protected_links:
                raise RuntimeError(
                    f"duplicate artifact-schema slot intersects protected collection: artifact={artifact_id} slot={slot}"
                )
            rows = connection.execute(
                """select id, artifact_id, schema_id, slot, feature_ref_is_semantic,
                run_id, created_by_id, created_at from lamindb_artifactschema
                where artifact_id = ? and slot = ? order by id""",
                (artifact_id, slot),
            ).fetchall()
            features = {
                row["schema_id"]: {
                    feature[0]
                    for feature in connection.execute(
                        "select feature_id from lamindb_schemafeature where schema_id = ?",
                        (row["schema_id"],),
                    )
                }
                for row in rows
            }
            first, second = rows
            first_features = features[first["schema_id"]]
            second_features = features[second["schema_id"]]
            if first_features < second_features:
                shadow, keeper = first, second
            elif second_features < first_features:
                shadow, keeper = second, first
            else:
                raise RuntimeError(
                    f"duplicate artifact-schema slot schemas are not a strict feature subset: artifact={artifact_id} slot={slot}"
                )
            cursor = connection.execute(
                "update lamindb_artifactschema set slot = null where id = ? and slot = ?",
                (shadow["id"], slot),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"failed to canonicalize artifact-schema shadow row {shadow['id']}"
                )
            groups.append(
                {
                    "artifact_id": artifact_id,
                    "slot": slot,
                    "keeper_link_id": keeper["id"],
                    "keeper_schema_id": keeper["schema_id"],
                    "keeper_feature_ids": sorted(features[keeper["schema_id"]]),
                    "shadow_link_id": shadow["id"],
                    "shadow_schema_id": shadow["schema_id"],
                    "shadow_feature_ids": sorted(features[shadow["schema_id"]]),
                    "strict_superset_feature_ids": sorted(
                        features[keeper["schema_id"]] - features[shadow["schema_id"]]
                    ),
                    "rows_before": [
                        {field: row[field] for field in row_fields} for row in rows
                    ],
                    "shadow_slot_after": None,
                }
            )
        remaining = connection.execute(
            """select artifact_id, slot, count(*) from lamindb_artifactschema
            where slot is not null group by artifact_id, slot having count(*) > 1"""
        ).fetchall()
        if remaining:
            raise RuntimeError(
                f"duplicate artifact-schema slots remain after canonicalization: {remaining}"
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    return {
        "mode": "candidate-only-strict-subset-artifact-schema-slot-null",
        "protected_collection_key": protected_collection_key,
        "protected_collection_id": protected_collection_id,
        "duplicate_group_count": len(groups),
        "duplicate_row_count": 2 * len(groups),
        "canonicalized_shadow_count": len(groups),
        "groups": groups,
    }


def migration_compatible_bionty_encode_uid(
    upstream_encoder, *, registry, kwargs: dict
) -> dict:
    """Adapt Bionty's Source UID encoder to a Django historical model.

    Bionty migration 0050 passes ``apps.get_model('bionty', 'Source')`` to the
    current encoder. Django historical models intentionally omit custom class
    methods, while the encoder requires ``__get_name_with_module__``. Add only
    that exact identity method for the duration of the upstream call so the
    migration retains Bionty's own deterministic UID algorithm.
    """
    if hasattr(registry, "__get_name_with_module__"):
        return upstream_encoder(registry=registry, kwargs=kwargs)
    meta = getattr(registry, "_meta", None)
    if (
        getattr(meta, "app_label", None) != "bionty"
        or getattr(meta, "object_name", None) != "Source"
    ):
        raise RuntimeError(
            "unexpected historical registry requested Bionty UID compatibility"
        )
    setattr(
        registry,
        "__get_name_with_module__",
        classmethod(lambda cls: "bionty.Source"),
    )
    try:
        return upstream_encoder(registry=registry, kwargs=kwargs)
    finally:
        delattr(registry, "__get_name_with_module__")


def deploy_offline(candidate: Path) -> None:
    # Construct a local-only instance explicitly. This avoids importing lamindb,
    # loading user credentials, refreshing a hub token, or syncing cloud SQLite.
    import bionty._biorecord as bionty_biorecord
    from django.core.management import call_command
    from django.db import connections
    from lamindb_setup._check_setup import disable_auto_connect
    from lamindb_setup.core._settings import settings
    from lamindb_setup.core._settings_instance import InstanceSettings
    from lamindb_setup.core.django import setup_django

    upstream_encoder = bionty_biorecord.encode_uid

    def compatible_encoder(*, registry, kwargs):
        return migration_compatible_bionty_encode_uid(
            upstream_encoder, registry=registry, kwargs=kwargs
        )

    bionty_biorecord.encode_uid = compatible_encoder
    try:
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
    finally:
        bionty_biorecord.encode_uid = upstream_encoder
        connections.close_all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("original", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("expected_original_sha256")
    parser.add_argument("registry_manifest", type=Path)
    parser.add_argument("membership_contract", type=Path)
    args = parser.parse_args()

    original = args.original.resolve()
    candidate = args.candidate.resolve()
    receipt = args.receipt.resolve()
    registry_manifest = args.registry_manifest.resolve()
    membership_contract = args.membership_contract.resolve()
    if not original.is_file() or original.stat().st_size == 0:
        raise FileNotFoundError(original)
    if candidate.exists():
        raise FileExistsError(candidate)
    if receipt.exists():
        raise FileExistsError(receipt)
    if not registry_manifest.is_file() or not membership_contract.is_file():
        raise FileNotFoundError([str(registry_manifest), str(membership_contract)])

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

    duplicate_hash_audit = canonicalize_duplicate_artifact_hashes(
        candidate, EXPECTED_COLLECTION_KEY
    )
    artifact_schema_slot_audit = canonicalize_duplicate_artifact_schema_slots(
        candidate, EXPECTED_COLLECTION_KEY
    )
    deploy_offline(candidate)
    after = snapshot(candidate)
    if after["integrity"] != "ok":
        raise RuntimeError(f"candidate integrity check failed: {after['integrity']}")
    changed_counts = {
        name: {"before": count, "after": after["counts"].get(name)}
        for name, count in before["counts"].items()
        if after["counts"].get(name) != count
    }
    if changed_counts:
        raise RuntimeError(
            f"pre-existing LaminDB/Bionty row counts changed: {changed_counts}"
        )
    if "lamindb_branch" not in after["tables"]:
        raise RuntimeError("migration did not add lamindb_branch")
    if len(after["tables"]) <= len(before["tables"]):
        raise RuntimeError("migration did not increase the table count")
    if after["migrations"] <= before["migrations"]:
        raise RuntimeError("migration did not advance django_migrations")

    from verify_migrated_lamindb import collection_membership

    registry = json.loads(registry_manifest.read_text())
    contract = json.loads(membership_contract.read_text())
    expected_count = contract["filtered_artifact_count"]
    expected_digest = contract["filtered_artifact_uids_sha256"]
    expected_cells = contract["filtered_cell_count"]
    if contract["collection"] != EXPECTED_COLLECTION_KEY:
        raise RuntimeError("membership contract collection mismatch")
    if (
        registry["kept_count"] != expected_count
        or registry["kept_uids_sha256"] != expected_digest
        or registry["inferred_cells"] != expected_cells
    ):
        raise RuntimeError("registry and membership contracts disagree")
    before_membership = collection_membership(original, EXPECTED_COLLECTION_KEY)
    after_membership = collection_membership(candidate, EXPECTED_COLLECTION_KEY)
    for observed in (before_membership, after_membership):
        if (
            observed["collection_id"] != registry["filtered_collection_id"]
            or observed["artifact_count"] != expected_count
            or observed["unique_artifact_count"] != expected_count
            or observed["link_count"] != expected_count
            or observed["artifact_uids_sha256"] != expected_digest
        ):
            raise RuntimeError("filtered collection identity/membership mismatch")
    if before_membership != after_membership:
        raise RuntimeError("filtered collection membership changed during migration")
    public_membership = dict(after_membership)
    public_membership.pop("artifact_uids")

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
        "before_data_table_counts": before["counts"],
        "candidate_data_table_counts": after["counts"],
        "membership": public_membership,
        "duplicate_hash_canonicalization": duplicate_hash_audit,
        "artifact_schema_slot_canonicalization": artifact_schema_slot_audit,
        "filtered_cell_count": expected_cells,
        "registry_manifest": str(registry_manifest),
        "registry_manifest_sha256": sha256(registry_manifest),
        "membership_contract": str(membership_contract),
        "membership_contract_sha256": sha256(membership_contract),
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
