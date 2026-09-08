from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scprint2.evaluation import scvi_baselines

REPO_ROOT = Path(__file__).parents[1]


def test_task3_paper_scvi_version_check_accepts_archived_stack(monkeypatch):
    monkeypatch.setattr(
        scvi_baselines,
        "package_versions",
        lambda packages: {
            package: scvi_baselines.TASK3_PAPER_SCVI_EXPECTED_VERSIONS[package]
            for package in packages
        },
    )

    assert (
        scvi_baselines.task3_paper_scvi_versions()
        == scvi_baselines.TASK3_PAPER_SCVI_EXPECTED_VERSIONS
    )


def test_task3_paper_scvi_version_check_rejects_modern_stack(monkeypatch):
    versions = dict(scvi_baselines.TASK3_PAPER_SCVI_EXPECTED_VERSIONS)
    versions["scvi-tools"] = "1.3.0"
    monkeypatch.setattr(
        scvi_baselines,
        "package_versions",
        lambda packages: {package: versions[package] for package in packages},
    )

    with pytest.raises(RuntimeError, match="Paper scVI environment mismatch"):
        scvi_baselines.task3_paper_scvi_versions()


def test_write_json_metadata_is_deterministic(tmp_path: Path):
    output = tmp_path / "artifact.metadata.json"

    scvi_baselines.write_json_metadata(output, {"z": 1, "a": {"b": 2}})

    assert json.loads(output.read_text()) == {"a": {"b": 2}, "z": 1}
    assert output.read_text().endswith("\n")


def test_scvi_wrappers_keep_help_independent_of_scvi_installation():
    for script in (
        "scripts/run_openproblems_scvi.py",
        "scripts/run_task3_paper_scvi.py",
    ):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / script), "--help"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "--input" in result.stdout
        assert "--output" in result.stdout
