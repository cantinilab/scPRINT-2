#!/usr/bin/env python3
"""Finish and adopt an interrupted primary cache without rebuilding its nine files."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXISTING_CACHE_FILES = (
    "categories",
    "idx_full.npy",
    "klass_indices.pt",
    "klass_indices_offsets.pt",
    "nnz.npy",
    "test_datasets.txt",
    "test_idx.npy",
    "train_labels.npy",
    "valid_idx.npy",
)
TRAIN_WEIGHTS = "train_weights.npy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_cache(
    cache: Path, names: Iterable[str]
) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for name in names:
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    return result


def assert_snapshot_unchanged(before: dict, after: dict) -> None:
    if before != after:
        changed = sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
        )
        raise RuntimeError(
            f"pre-existing cache files mutated during adoption: {changed}"
        )


def derive_train_weights(
    labels: np.ndarray, weight_scaler: float, chunk_size: int = 10_000_000
) -> np.ndarray:
    """Derive the current LabelWeightedSampler class weights in bounded memory."""
    max_label = int(labels.max())
    counts = np.zeros(max_label + 1, dtype=np.int64)
    for start in range(0, len(labels), chunk_size):
        chunk = np.asarray(labels[start : start + chunk_size], dtype=np.int64)
        counts += np.bincount(chunk, minlength=max_label + 1)
    counts[-1] = 0
    weights = (float(weight_scaler) * counts) / (counts + float(weight_scaler))
    return weights.astype(np.float32)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def accepted_datamodule_parameters(DataModule: type, DataLoader: type) -> set[str]:
    explicit = {
        name
        for name, parameter in inspect.signature(DataModule.__init__).parameters.items()
        if name != "self"
        and parameter.kind not in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}
    }
    dataloader = {
        name
        for name, parameter in inspect.signature(DataLoader.__init__).parameters.items()
        if name != "self"
        and parameter.kind not in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}
    }
    return explicit | (
        dataloader - {"dataset", "sampler", "batch_sampler", "collate_fn"}
    )


def batch_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in batch.items()
    }


def main() -> None:
    import torch
    import yaml
    from scdataloader import DataModule
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("membership", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("ntv3", type=Path)
    parser.add_argument("registry_manifest", type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)

    before = snapshot_cache(args.cache, EXISTING_CACHE_FILES)
    config = yaml.safe_load(args.config.read_text())
    membership = json.loads(args.membership.read_text())
    data = dict(config["data"])
    if data["collection_name"] != membership["collection"]:
        raise ValueError("config and membership collection identities disagree")
    expected = {
        "n_samples": membership["filtered_cell_count"],
        "valid": 17_377_896,
        "test": 6_679_542,
        "predict": 323_500_490,
        "sampler_classes": membership["expected_sampler_classes"],
        "artifacts": membership["filtered_artifact_count"],
    }

    labels = np.load(args.cache / "train_labels.npy", mmap_mode="r")
    weights = derive_train_weights(labels, data["weight_scaler"])
    if len(weights) != expected["sampler_classes"]:
        raise RuntimeError(
            f"train_weights cardinality mismatch: {len(weights)} != {expected['sampler_classes']}"
        )
    weight_path = args.cache / TRAIN_WEIGHTS
    if weight_path.exists():
        existing = np.load(weight_path, mmap_mode="r")
        if (
            existing.dtype != np.float32
            or existing.shape != weights.shape
            or not np.array_equal(existing, weights)
        ):
            raise RuntimeError(
                "existing train_weights does not match current sampler formula"
            )
    else:
        atomic_save_npy(weight_path, weights)

    data.update(
        store_location=str(args.cache),
        gene_embeddings=str(args.ntv3),
        force_recompute_indices=False,
        sampler_workers=44,
    )
    kwargs = {
        key: value
        for key, value in data.items()
        if key in accepted_datamodule_parameters(DataModule, DataLoader)
    }
    datamodule = DataModule(**kwargs)
    datamodule.setup("fit")
    train = datamodule.train_dataloader()
    validation = datamodule.val_dataloader()
    datamodule.test_dataloader()
    datamodule.predict_dataloader()
    train_batch = next(iter(train))
    validation_batch = next(iter(validation))
    sampler = train.sampler
    actual = {
        "n_samples": datamodule.n_samples,
        "valid": len(datamodule.valid_idx),
        "test": len(datamodule.test_idx),
        "predict": len(datamodule.idx_full),
        "sampler_classes": len(sampler.klass_offsets),
        "artifacts": len(datamodule.dataset.mapped_dataset.storages),
    }
    if actual != expected:
        raise RuntimeError(
            f"cache identity mismatch: actual={actual}, expected={expected}"
        )
    categories = torch.load(
        args.cache / "categories", map_location="cpu", weights_only=False
    )
    category_lengths = {key: len(value) for key, value in categories.items()}
    if set(category_lengths.values()) != {expected["artifacts"]}:
        raise RuntimeError(f"category length mismatch: {category_lengths}")
    after = snapshot_cache(args.cache, EXISTING_CACHE_FILES)
    assert_snapshot_unchanged(before, after)
    all_files = snapshot_cache(args.cache, (*EXISTING_CACHE_FILES, TRAIN_WEIGHTS))
    receipt = {
        "status": "accepted",
        "adoption": "nine-existing-files-plus-derived-train-weights-no-rebuild",
        "expected": expected,
        "actual": actual,
        "classes": datamodule.classes,
        "category_lengths": category_lengths,
        "train_batch": batch_shapes(train_batch),
        "validation_batch": batch_shapes(validation_batch),
        "train_weights": {
            "cardinality": len(weights),
            "dtype": str(weights.dtype),
            "min": float(weights.min()),
            "max": float(weights.max()),
        },
        "cache": str(args.cache.resolve()),
        "cache_files": [
            {"path": name, **metadata} for name, metadata in all_files.items()
        ],
        "cache_bytes": sum(int(metadata["size"]) for metadata in all_files.values()),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256(args.config),
        "membership": membership,
        "membership_sha256": sha256(args.membership),
        "ntv3": str(args.ntv3.resolve()),
        "ntv3_sha256": sha256(args.ntv3),
        "registry_manifest": json.loads(args.registry_manifest.read_text()),
        "effective_data": data,
    }
    atomic_json(args.receipt, receipt)
    print("PRIMARY_FILTERED_CACHE_ADOPTION_PASS " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
