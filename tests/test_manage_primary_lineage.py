from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def load_manager():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "training"
        / "manage_primary_lineage.py"
    )
    spec = importlib.util.spec_from_file_location("manage_primary_lineage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setenv("EXPECTED_SCPRINT_COMMIT", "scprint-final")
    monkeypatch.setenv("EXPECTED_SCDATALOADER_COMMIT", "scd-final")
    return load_manager()


def begin(manager, root, active, segment, run_id, now):
    return manager.begin_segment(
        root, active, "job-1", segment, "logical-group", run_id, now=now
    )


def end(manager, root, segment, reason, now):
    return manager.end_segment(root, "job-1", segment, reason, now=now)


def test_fresh_segment_zero_requires_absent_root_and_active(manager, tmp_path):
    root = tmp_path / "run"
    active = tmp_path / "active.json"
    result = begin(manager, root, active, 0, "wandb-0", 100)
    assert result["random_init"] is True
    assert result["checkpoint"] is None

    with pytest.raises(AssertionError):
        begin(manager, root, tmp_path / "other-active.json", 0, "wandb-new", 101)
    with pytest.raises(AssertionError):
        begin(manager, tmp_path / "other-root", active, 0, "wandb-new", 101)


def test_segments_require_contiguous_unique_checkpointed_restarts(manager, tmp_path):
    root = tmp_path / "run"
    active = tmp_path / "active.json"
    begin(manager, root, active, 0, "wandb-0", 100)

    with pytest.raises(AssertionError):
        begin(manager, root, active, 1, "wandb-1", 101)
    end(manager, root, 0, "native-requeue-checkpoint-1", 120)
    with pytest.raises(AssertionError):
        begin(manager, root, active, 2, "wandb-2", 121)
    with pytest.raises(AssertionError):
        begin(manager, root, active, 1, "wandb-1", 121)

    checkpoint = root / "hpc_ckpt_1.ckpt"
    checkpoint.write_bytes(b"")
    with pytest.raises(AssertionError):
        begin(manager, root, active, 1, "wandb-1", 121)
    checkpoint.write_bytes(b"checkpoint-1")
    with pytest.raises(AssertionError):
        begin(manager, root, active, 1, "wandb-0", 121)

    resumed = begin(manager, root, active, 1, "wandb-1", 121)
    assert resumed["random_init"] is False
    assert resumed["checkpoint"] == str(checkpoint.resolve())
    with pytest.raises(AssertionError):
        begin(manager, root, active, 1, "wandb-duplicate", 122)


def test_elapsed_budget_accumulates_and_rejects_overrun(manager, tmp_path):
    root = tmp_path / "run"
    active = tmp_path / "active.json"
    begin(manager, root, active, 0, "wandb-0", 100)
    ledger = end(manager, root, 0, "native-requeue-checkpoint-1", 200)
    assert ledger["cumulative_elapsed_seconds"] == 100

    (root / "hpc_ckpt_1.ckpt").write_bytes(b"checkpoint-1")
    begin(manager, root, active, 1, "wandb-1", 200)
    with pytest.raises(AssertionError):
        end(manager, root, 1, "over-budget", 200 + manager.BUDGET_SECONDS + 3601)


def test_finalize_requires_exactly_five_ended_segments_and_blocks_sixth(
    manager, tmp_path
):
    root = tmp_path / "run"
    active = tmp_path / "active.json"
    begin(manager, root, active, 0, "wandb-0", 0)
    with pytest.raises(AssertionError):
        manager.finalize_lineage(root, active, now=1)

    for segment in range(5):
        if segment:
            (root / f"hpc_ckpt_{segment}.ckpt").write_bytes(
                f"checkpoint-{segment}".encode()
            )
            begin(manager, root, active, segment, f"wandb-{segment}", segment * 10)
        end(
            manager,
            root,
            segment,
            f"native-requeue-checkpoint-{segment + 1}",
            segment * 10 + 5,
        )

    final = manager.finalize_lineage(root, active, now=60)
    assert final["status"] == "terminal_100h_complete"
    active_receipt = json.loads(active.read_text())
    assert active_receipt["status"] == "terminal_100h_complete"
    assert active_receipt["segments"] == 5
    with pytest.raises(AssertionError):
        begin(manager, root, active, 5, "wandb-5", 61)
