from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "config" / "base_v3_ntv3_primary.yml"
MEMBERSHIP = ROOT / "config" / "base_v3_ntv3_primary_membership.json"


def test_primary_config_preserves_reference_recipe_without_cluster_paths():
    config = yaml.safe_load(CONFIG.read_text())

    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["log_every_n_steps"] == 100
    assert config["wandblog"] == ""
    assert config["model"]["attention"] == "normal"
    assert config["model"]["dropout"] == 0.02
    assert config["model"]["expr_encoder_layers"] == 1
    assert config["model"]["layers_cls"] == [256]
    assert config["model"]["precpt_gene_emb"] is None
    assert config["model"]["gene_pos_file"] is None
    assert config["scprint_training"]["noise"] == [0.8, 1.0]
    assert config["scprint_training"]["mask_ratio"] == []
    assert config["scprint_training"]["vae_kl_scale"] == 0.0002
    assert config["data"]["collection_name"].endswith(" filtered")
    assert config["data"]["max_len"] == 3200
    assert config["data"]["weight_scaler"] == 500
    assert config["data"]["batch_size"] == 50
    assert config["data"]["n_samples_per_epoch"] == 1_000_000
    assert "/lustre/" not in CONFIG.read_text()


def test_filtered_membership_matches_historical_counts_and_digest():
    membership = json.loads(MEMBERSHIP.read_text())
    excluded = membership["excluded_artifacts"]

    assert membership["source_artifact_count"] == 26_337
    assert membership["filtered_artifact_count"] == 26_332
    assert membership["filtered_cell_count"] == 347_557_928
    assert membership["expected_sampler_classes"] == 138_734
    assert len(excluded) == 5
    assert sum(item["n_obs"] for item in excluded) == 97_353
    assert [item["uid"] for item in excluded] == sorted(
        item["uid"] for item in excluded
    )
    assert (
        membership["filtered_artifact_uids_sha256"]
        == "50443e22451aaad1c317a9866b86b1b0f2b5d6c27fc4d394e2357a73457a6482"
    )


def test_cache_builder_uses_reviewed_config_instead_of_redefining_recipe():
    script_path = ROOT / "scripts" / "training" / "prepare_primary_filtered_cache.py"
    spec = importlib.util.spec_from_file_location("prepare_primary_filtered_cache", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.sha256(MEMBERSHIP) == hashlib.sha256(MEMBERSHIP.read_bytes()).hexdigest()
    parameters = module.accepted_datamodule_parameters()
    for name in (
        "collection_name",
        "store_location",
        "weight_scaler",
        "batch_size",
        "n_samples_per_epoch",
    ):
        assert name in parameters
