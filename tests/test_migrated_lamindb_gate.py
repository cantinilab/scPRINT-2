from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLLECTION_KEY = "scPRINT-V2 (all+tahoe+scbase) filtered"


def load_verifier():
    path = ROOT / "scripts" / "training" / "verify_migrated_lamindb.py"
    spec = importlib.util.spec_from_file_location("verify_migrated_lamindb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_migrator():
    path = ROOT / "scripts" / "training" / "migrate_lamindb2_offline.py"
    spec = importlib.util.spec_from_file_location("migrate_lamindb2_offline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_contract(tmp_path: Path) -> dict[str, Path]:
    original = tmp_path / "original.lndb"
    connection = sqlite3.connect(original)
    connection.execute(
        "create table lamindb_artifact (id integer primary key, uid text, n_observations integer)"
    )
    connection.executemany(
        "insert into lamindb_artifact values (?, ?, ?)",
        [(1, "artifact-a", 50), (2, "artifact-b", 73)],
    )
    connection.execute(
        "create table lamindb_collection (id integer primary key, uid text, key text)"
    )
    connection.execute(
        "insert into lamindb_collection values (?, ?, ?)",
        (29, "collection-uid", COLLECTION_KEY),
    )
    connection.execute(
        "create table lamindb_collectionartifact (id integer primary key, artifact_id integer, collection_id integer)"
    )
    connection.executemany(
        "insert into lamindb_collectionartifact values (?, ?, 29)", [(1, 1), (2, 2)]
    )
    connection.execute("create table lamindb_user (id integer primary key)")
    connection.execute("insert into lamindb_user values (1)")
    connection.execute("create table bionty_gene (id integer primary key)")
    connection.execute("insert into bionty_gene values (1)")
    connection.execute(
        "create table django_migrations (id integer primary key, app text, name text)"
    )
    connection.execute("insert into django_migrations values (1, 'lamindb', '0001')")
    connection.commit()
    connection.close()

    candidate = tmp_path / "candidate.lndb"
    shutil.copy2(original, candidate)
    connection = sqlite3.connect(candidate)
    connection.execute("create table lamindb_branch (id integer primary key)")
    connection.execute("insert into django_migrations values (2, 'lamindb', '0002')")
    connection.commit()
    connection.close()

    uids = ["artifact-a", "artifact-b"]
    uid_digest = hashlib.sha256(("\n".join(uids) + "\n").encode()).hexdigest()
    membership = tmp_path / "membership.json"
    membership.write_text(
        json.dumps(
            {
                "collection": COLLECTION_KEY,
                "filtered_artifact_count": 2,
                "filtered_cell_count": 123,
                "filtered_artifact_uids_sha256": uid_digest,
                "excluded_artifacts": [],
            }
        )
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "output": str(original.resolve()),
                "output_sha256": sha256(original),
                "filtered_collection_id": 29,
                "kept_count": 2,
                "kept_uids_sha256": uid_digest,
                "inferred_cells": 123,
            }
        )
    )
    migration = tmp_path / "migration.json"
    migration.write_text(
        json.dumps(
            {
                "status": "accepted_candidate",
                "swapped": False,
                "original": str(original.resolve()),
                "candidate": str(candidate.resolve()),
                "original_sha256": sha256(original),
                "candidate_sha256": sha256(candidate),
                "duplicate_hash_canonicalization": {
                    "mode": "candidate-only-exact-identity-shadow-hash-null",
                    "protected_collection_key": COLLECTION_KEY,
                    "protected_collection_id": 29,
                    "ranking": [
                        "collection_link_count_desc",
                        "is_latest_desc",
                        "active_branch_desc",
                        "id_asc",
                    ],
                    "duplicate_group_count": 0,
                    "duplicate_row_count": 0,
                    "canonicalized_shadow_count": 0,
                    "groups": [],
                },
                "artifact_schema_slot_canonicalization": {
                    "mode": "candidate-only-strict-subset-artifact-schema-slot-null",
                    "protected_collection_key": COLLECTION_KEY,
                    "duplicate_group_count": 0,
                    "duplicate_row_count": 0,
                    "canonicalized_shadow_count": 0,
                    "groups": [],
                },
            }
        )
    )
    return {
        "original": original,
        "candidate": candidate,
        "membership": membership,
        "registry": registry,
        "migration": migration,
        "verification": tmp_path / "verification.json",
    }


def test_create_and_verify_exact_migration_receipt(tmp_path):
    module = load_verifier()
    paths = make_contract(tmp_path)

    created = module.create_receipt(
        paths["original"],
        paths["candidate"],
        paths["migration"],
        paths["registry"],
        paths["membership"],
        paths["verification"],
    )
    verified = module.verify(paths["verification"], paths["candidate"])

    assert created["status"] == "accepted_migrated_lamindb"
    assert created["filtered_cell_count"] == 123
    assert created["membership"]["artifact_count"] == 2
    assert created["before_data_table_counts"] == {
        "bionty_gene": 1,
        "lamindb_artifact": 2,
        "lamindb_collection": 1,
        "lamindb_collectionartifact": 2,
        "lamindb_user": 1,
    }
    assert verified["status"] == "accepted"
    assert verified["database_sha256"] == sha256(paths["candidate"])


def test_create_rejects_changed_preexisting_link_or_data_table(tmp_path):
    module = load_verifier()
    paths = make_contract(tmp_path)
    connection = sqlite3.connect(paths["candidate"])
    connection.execute("insert into bionty_gene values (2)")
    connection.commit()
    connection.close()
    migration = json.loads(paths["migration"].read_text())
    migration["candidate_sha256"] = sha256(paths["candidate"])
    paths["migration"].write_text(json.dumps(migration))

    with pytest.raises(RuntimeError, match="table parity"):
        module.create_receipt(
            paths["original"],
            paths["candidate"],
            paths["migration"],
            paths["registry"],
            paths["membership"],
            paths["verification"],
        )


def test_verify_rejects_candidate_mutated_after_receipt(tmp_path):
    module = load_verifier()
    paths = make_contract(tmp_path)
    module.create_receipt(
        paths["original"],
        paths["candidate"],
        paths["migration"],
        paths["registry"],
        paths["membership"],
        paths["verification"],
    )
    connection = sqlite3.connect(paths["candidate"])
    connection.execute("delete from lamindb_collectionartifact where id = 2")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="SHA-256"):
        module.verify(paths["verification"], paths["candidate"])


def test_launchers_require_independent_migration_verification():
    for name in (
        "scprint2_r3_cache_adopt.sbatch",
        "scprint2_r3_primary_smoke.sbatch",
        "scprint2_r3_primary_long.sbatch",
    ):
        source = (ROOT / "slurm" / name).read_text()
        assert "LAMINDB_DATABASE_PATH" in source
        assert "LAMINDB_VERIFICATION_RECEIPT" in source
        assert "verify_migrated_lamindb.py\" verify" in source
        assert "DB=$R/scprint2_base_v3_filt_esi_t_6cff2510.lndb" not in source


def test_migration_helper_checks_every_preexisting_data_table_and_membership():
    source = (
        ROOT / "scripts" / "training" / "migrate_lamindb2_offline.py"
    ).read_text()
    assert 'DATA_PREFIXES = ("lamindb_", "bionty_")' in source
    assert "for name, count in before[\"counts\"].items()" in source
    assert "collection_membership" in source
    assert "filtered_cell_count" in source
    launcher = (ROOT / "slurm" / "scprint2_r3_migrate_lamindb.sbatch").read_text()
    assert "REGISTRY_MANIFEST" in launcher
    assert "MEMBERSHIP_CONTRACT" in launcher
    assert "EXPECTED_ORIGINAL_SHA256" in launcher


def make_duplicate_hash_database(tmp_path: Path) -> Path:
    database = tmp_path / "duplicate-hashes.lndb"
    connection = sqlite3.connect(database)
    connection.execute(
        """create table lamindb_artifact (
        id integer primary key, uid text not null, hash text, key text,
        size integer, _hash_type text, is_latest bool not null, _branch_code integer not null
        )"""
    )
    connection.executemany(
        "insert into lamindb_artifact values (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "uid-1", "hash-1", "key-1", 10, "md5-n", 1, -1),
            (2, "uid-2", "hash-1", "key-1", 10, "md5-n", 0, 1),
            (3, "uid-3", "hash-2", "key-2", 20, "md5-n", 1, -1),
            (4, "uid-4", "hash-2", "key-2", 20, "md5-n", 1, 1),
            (5, "uid-5", "hash-3", "key-3", 30, "md5-n", 1, 1),
            (6, "uid-6", "hash-3", "key-3", 30, "md5-n", 1, 1),
            (7, "filtered", "unique", "filtered", 40, "md5-n", 1, 1),
        ],
    )
    connection.execute(
        "create table lamindb_collection (id integer primary key, uid text, key text)"
    )
    connection.execute(
        "insert into lamindb_collection values (29, 'collection-uid', ?)",
        (COLLECTION_KEY,),
    )
    connection.execute(
        "create table lamindb_collectionartifact (id integer primary key, artifact_id integer, collection_id integer)"
    )
    connection.executemany(
        "insert into lamindb_collectionartifact values (?, ?, ?)",
        [(1, 2, 30), (2, 7, 29)],
    )
    connection.commit()
    connection.close()
    return database


def test_candidate_duplicate_hash_canonicalization_is_deterministic_and_audited(tmp_path):
    module = load_migrator()
    database = make_duplicate_hash_database(tmp_path)

    audit = module.canonicalize_duplicate_artifact_hashes(database, COLLECTION_KEY)

    assert audit["duplicate_group_count"] == 3
    assert audit["duplicate_row_count"] == 6
    assert audit["canonicalized_shadow_count"] == 3
    assert audit["ranking"] == [
        "collection_link_count_desc",
        "is_latest_desc",
        "active_branch_desc",
        "id_asc",
    ]
    assert [(item["keeper_id"], item["shadow_id"]) for item in audit["groups"]] == [
        (2, 1),
        (4, 3),
        (5, 6),
    ]
    connection = sqlite3.connect(database)
    observed = connection.execute(
        "select id, hash from lamindb_artifact where id <= 6 order by id"
    ).fetchall()
    protected = connection.execute(
        "select artifact_id from lamindb_collectionartifact where collection_id = 29"
    ).fetchall()
    connection.close()
    assert observed == [
        (1, None),
        (2, "hash-1"),
        (3, None),
        (4, "hash-2"),
        (5, "hash-3"),
        (6, None),
    ]
    assert protected == [(7,)]


def test_candidate_duplicate_hash_canonicalization_rejects_nonidentical_identity(tmp_path):
    module = load_migrator()
    database = make_duplicate_hash_database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("update lamindb_artifact set size = 11 where id = 2")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="content identity"):
        module.canonicalize_duplicate_artifact_hashes(database, COLLECTION_KEY)


def test_candidate_duplicate_hash_canonicalization_rejects_protected_membership(tmp_path):
    module = load_migrator()
    database = make_duplicate_hash_database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "insert into lamindb_collectionartifact values (3, 1, 29)"
    )
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="protected collection"):
        module.canonicalize_duplicate_artifact_hashes(database, COLLECTION_KEY)


def test_independent_verifier_replays_duplicate_hash_audit(tmp_path):
    migrator = load_migrator()
    verifier = load_verifier()
    original = make_duplicate_hash_database(tmp_path)
    candidate = tmp_path / "candidate.lndb"
    shutil.copy2(original, candidate)
    audit = migrator.canonicalize_duplicate_artifact_hashes(candidate, COLLECTION_KEY)

    verified = verifier.verify_duplicate_hash_canonicalization(
        original, candidate, audit, COLLECTION_KEY
    )

    assert verified["duplicate_group_count"] == 3
    assert verified["canonicalized_shadow_count"] == 3


def test_independent_verifier_rejects_unrecorded_shadow_hash(tmp_path):
    migrator = load_migrator()
    verifier = load_verifier()
    original = make_duplicate_hash_database(tmp_path)
    candidate = tmp_path / "candidate.lndb"
    shutil.copy2(original, candidate)
    audit = migrator.canonicalize_duplicate_artifact_hashes(candidate, COLLECTION_KEY)
    connection = sqlite3.connect(candidate)
    connection.execute("update lamindb_artifact set hash = 'tampered' where id = 1")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="shadow hash"):
        verifier.verify_duplicate_hash_canonicalization(
            original, candidate, audit, COLLECTION_KEY
        )


def make_duplicate_artifact_schema_database(tmp_path: Path) -> Path:
    database = tmp_path / "duplicate-artifact-schema-slots.lndb"
    connection = sqlite3.connect(database)
    connection.execute(
        "create table lamindb_artifact (id integer primary key, uid text not null)"
    )
    connection.executemany(
        "insert into lamindb_artifact values (?, ?)",
        [(1, "artifact-1"), (2, "artifact-2"), (3, "filtered")],
    )
    connection.execute(
        "create table lamindb_collection (id integer primary key, uid text, key text)"
    )
    connection.execute(
        "insert into lamindb_collection values (29, 'collection-uid', ?)",
        (COLLECTION_KEY,),
    )
    connection.execute(
        "create table lamindb_collectionartifact (id integer primary key, artifact_id integer, collection_id integer)"
    )
    connection.execute("insert into lamindb_collectionartifact values (1, 3, 29)")
    connection.execute(
        "create table lamindb_schema (id integer primary key, uid text, n integer)"
    )
    connection.executemany(
        "insert into lamindb_schema values (?, ?, ?)",
        [(14, "schema-subset", 2), (181, "schema-superset", 3), (200, "other", 1)],
    )
    connection.execute(
        "create table lamindb_schemafeature (id integer primary key, schema_id integer, feature_id integer)"
    )
    connection.executemany(
        "insert into lamindb_schemafeature values (?, ?, ?)",
        [(1, 14, 5), (2, 14, 6), (3, 181, 5), (4, 181, 6), (5, 181, 7), (6, 200, 9)],
    )
    connection.execute(
        """create table lamindb_artifactschema (
        id integer primary key, artifact_id integer not null, schema_id integer not null,
        slot text, feature_ref_is_semantic bool, run_id integer, created_by_id integer,
        created_at text
        )"""
    )
    connection.executemany(
        "insert into lamindb_artifactschema values (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (10, 1, 14, "obs", 1, 4, 1, "2023-01-01"),
            (11, 1, 181, "obs", 1, 4, 1, "2023-01-02"),
            (12, 2, 14, "obs", 1, None, 1, "2023-01-01"),
            (13, 2, 181, "obs", 1, None, 1, "2023-01-02"),
            (14, 2, 200, "var", 0, None, 1, "2023-01-03"),
        ],
    )
    connection.commit()
    connection.close()
    return database


def test_candidate_artifact_schema_slot_canonicalization_keeps_strict_superset(tmp_path):
    module = load_migrator()
    database = make_duplicate_artifact_schema_database(tmp_path)

    audit = module.canonicalize_duplicate_artifact_schema_slots(database, COLLECTION_KEY)

    assert audit["duplicate_group_count"] == 2
    assert audit["duplicate_row_count"] == 4
    assert audit["canonicalized_shadow_count"] == 2
    assert [(group["keeper_schema_id"], group["shadow_schema_id"]) for group in audit["groups"]] == [
        (181, 14),
        (181, 14),
    ]
    connection = sqlite3.connect(database)
    observed = connection.execute(
        "select id, artifact_id, schema_id, slot from lamindb_artifactschema order by id"
    ).fetchall()
    connection.close()
    assert observed == [
        (10, 1, 14, None),
        (11, 1, 181, "obs"),
        (12, 2, 14, None),
        (13, 2, 181, "obs"),
        (14, 2, 200, "var"),
    ]


def test_candidate_artifact_schema_slot_canonicalization_rejects_non_subset(tmp_path):
    module = load_migrator()
    database = make_duplicate_artifact_schema_database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("delete from lamindb_schemafeature where schema_id = 181 and feature_id = 6")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="strict feature subset"):
        module.canonicalize_duplicate_artifact_schema_slots(database, COLLECTION_KEY)


def test_candidate_artifact_schema_slot_canonicalization_rejects_protected_membership(tmp_path):
    module = load_migrator()
    database = make_duplicate_artifact_schema_database(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("insert into lamindb_collectionartifact values (2, 1, 29)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="protected collection"):
        module.canonicalize_duplicate_artifact_schema_slots(database, COLLECTION_KEY)


def test_independent_verifier_replays_artifact_schema_slot_audit(tmp_path):
    migrator = load_migrator()
    verifier = load_verifier()
    original = make_duplicate_artifact_schema_database(tmp_path)
    candidate = tmp_path / "candidate-artifact-schema.lndb"
    shutil.copy2(original, candidate)
    audit = migrator.canonicalize_duplicate_artifact_schema_slots(candidate, COLLECTION_KEY)

    verified = verifier.verify_artifact_schema_slot_canonicalization(
        original, candidate, audit, COLLECTION_KEY
    )

    assert verified == {
        "duplicate_group_count": 2,
        "canonicalized_shadow_count": 2,
    }


def test_independent_verifier_rejects_unrecorded_artifact_schema_slot_change(tmp_path):
    migrator = load_migrator()
    verifier = load_verifier()
    original = make_duplicate_artifact_schema_database(tmp_path)
    candidate = tmp_path / "candidate-artifact-schema.lndb"
    shutil.copy2(original, candidate)
    audit = migrator.canonicalize_duplicate_artifact_schema_slots(candidate, COLLECTION_KEY)
    connection = sqlite3.connect(candidate)
    connection.execute("update lamindb_artifactschema set slot = 'tampered' where id = 10")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="shadow slot"):
        verifier.verify_artifact_schema_slot_canonicalization(
            original, candidate, audit, COLLECTION_KEY
        )


def test_independent_verifier_rejects_protected_artifact_schema_canonicalization(tmp_path):
    migrator = load_migrator()
    verifier = load_verifier()
    original = make_duplicate_artifact_schema_database(tmp_path)
    candidate = tmp_path / "candidate-artifact-schema.lndb"
    shutil.copy2(original, candidate)
    audit = migrator.canonicalize_duplicate_artifact_schema_slots(candidate, COLLECTION_KEY)
    for database in (original, candidate):
        connection = sqlite3.connect(database)
        connection.execute("insert into lamindb_collectionartifact values (2, 1, 29)")
        connection.commit()
        connection.close()

    with pytest.raises(RuntimeError, match="protected collection"):
        verifier.verify_artifact_schema_slot_canonicalization(
            original, candidate, audit, COLLECTION_KEY
        )


def test_historical_bionty_source_uid_compatibility_preserves_encoder_semantics():
    module = load_migrator()

    class Meta:
        app_label = "bionty"
        object_name = "Source"

    class HistoricalSource:
        _meta = Meta()

    def upstream_encoder(*, registry, kwargs):
        return {
            **kwargs,
            "uid": registry.__get_name_with_module__()
            + ":"
            + kwargs["entity"]
            + kwargs["name"]
            + kwargs["organism"]
            + kwargs["version"],
        }

    result = module.migration_compatible_bionty_encode_uid(
        upstream_encoder,
        registry=HistoricalSource,
        kwargs={
            "entity": "bionty.Gene",
            "name": "ensembl",
            "organism": "human",
            "version": "112",
        },
    )

    assert result["uid"] == "bionty.Source:bionty.Geneensemblhuman112"
    assert not hasattr(HistoricalSource, "__get_name_with_module__")


def test_historical_bionty_uid_compatibility_rejects_unexpected_registry():
    module = load_migrator()

    class Meta:
        app_label = "bionty"
        object_name = "Gene"

    class HistoricalGene:
        _meta = Meta()

    with pytest.raises(RuntimeError, match="unexpected historical registry"):
        module.migration_compatible_bionty_encode_uid(
            lambda **kwargs: kwargs,
            registry=HistoricalGene,
            kwargs={"name": "TP53"},
        )


def test_bionty_compatibility_is_installed_only_after_offline_django_setup():
    source = (
        ROOT / "scripts" / "training" / "migrate_lamindb2_offline.py"
    ).read_text()

    setup_position = source.index("disable_auto_connect(setup_django)(instance")
    bionty_import_position = source.index(
        'importlib.import_module("bionty._biorecord")'
    )

    assert setup_position < bionty_import_position


def test_migration_launcher_budgets_observed_full_schema_upgrade_runtime():
    source = (ROOT / "slurm" / "scprint2_r3_migrate_lamindb.sbatch").read_text()

    assert "#SBATCH --time=00:45:00" in source
