from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRESERVED = (
    "lamindb_artifact",
    "lamindb_collection",
    "lamindb_storage",
    "lamindb_transform",
    "lamindb_user",
)


def load_verifier():
    path = ROOT / "scripts" / "training" / "verify_migrated_lamindb.py"
    spec = importlib.util.spec_from_file_location("verify_migrated_lamindb", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_candidate(tmp_path: Path) -> tuple[Path, Path]:
    database = tmp_path / "candidate.lndb"
    connection = sqlite3.connect(database)
    for name in PRESERVED:
        connection.execute(f"create table {name} (id integer primary key)")
        connection.execute(f"insert into {name} values (1)")
    connection.execute("create table lamindb_branch (id integer primary key)")
    connection.execute(
        "create table django_migrations (id integer primary key, app text, name text)"
    )
    connection.execute("insert into django_migrations values (1, 'lamindb', '0001')")
    connection.commit()
    connection.close()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "accepted_candidate",
                "swapped": False,
                "candidate": str(database.resolve()),
                "candidate_sha256": sha256(database),
                "branch_table_added": True,
                "after_migrations": 1,
                "row_counts": {name: 1 for name in PRESERVED},
            }
        )
    )
    return database, receipt


def test_verifier_accepts_exact_unswapped_candidate(tmp_path):
    module = load_verifier()
    database, receipt = make_candidate(tmp_path)

    result = module.verify(receipt, database)

    assert result["status"] == "accepted"
    assert result["database_sha256"] == sha256(database)
    assert result["migrations"] == 1


def test_verifier_rejects_candidate_mutated_after_receipt(tmp_path):
    module = load_verifier()
    database, receipt = make_candidate(tmp_path)
    connection = sqlite3.connect(database)
    connection.execute("insert into lamindb_artifact values (2)")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="SHA-256"):
        module.verify(receipt, database)


def test_launchers_require_and_verify_migrated_database():
    for name in (
        "scprint2_r3_cache_adopt.sbatch",
        "scprint2_r3_primary_smoke.sbatch",
        "scprint2_r3_primary_long.sbatch",
    ):
        source = (ROOT / "slurm" / name).read_text()
        assert "LAMINDB_DATABASE_PATH" in source
        assert "LAMINDB_MIGRATION_RECEIPT" in source
        assert "verify_migrated_lamindb.py" in source
        assert "DB=$R/scprint2_base_v3_filt_esi_t_6cff2510.lndb" not in source
