#!/usr/bin/env python3
"""Build and verify the immutable primary-esi filtered training cache."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from scdataloader import DataModule
from torch.utils.data import DataLoader


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
        for key, value in batch.items()
    }


def accepted_datamodule_parameters() -> set[str]:
    explicit = {
        name
        for name, parameter in inspect.signature(DataModule.__init__).parameters.items()
        if name != "self"
        and parameter.kind
        not in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}
    }
    dataloader = {
        name
        for name, parameter in inspect.signature(DataLoader.__init__).parameters.items()
        if name != "self"
        and parameter.kind
        not in {parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL}
    }
    # DataModule forwards arbitrary DataLoader kwargs after setting its own sampler and
    # collator. Keep those runtime knobs while refusing unknown config keys.
    return explicit | (dataloader - {"dataset", "sampler", "batch_sampler", "collate_fn"})


def main() -> None:
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
    if args.cache.exists() and any(args.cache.iterdir()):
        raise FileExistsError(f"non-empty cache: {args.cache}")

    config = yaml.safe_load(args.config.read_text())
    membership = json.loads(args.membership.read_text())
    data = dict(config["data"])
    expected = {
        "n_samples": membership["filtered_cell_count"],
        "valid": 17_377_896,
        "test": 6_679_542,
        "predict": 323_500_490,
        "sampler_classes": membership["expected_sampler_classes"],
        "artifacts": membership["filtered_artifact_count"],
    }

    if data["collection_name"] != membership["collection"]:
        raise ValueError("config and membership collection identities disagree")
    data.update(
        store_location=str(args.cache),
        force_recompute_indices=False,
        sampler_workers=44,
    )
    kwargs = {
        key: value
        for key, value in data.items()
        if key in accepted_datamodule_parameters()
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
        raise RuntimeError(f"cache identity mismatch: actual={actual}, expected={expected}")

    categories = torch.load(
        args.cache / "categories", map_location="cpu", weights_only=False
    )
    category_lengths = {key: len(value) for key, value in categories.items()}
    if set(category_lengths.values()) != {expected["artifacts"]}:
        raise RuntimeError(f"category length mismatch: {category_lengths}")

    cache_files = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(args.cache.iterdir())
        if path.is_file()
    ]
    receipt = {
        "status": "accepted",
        "expected": expected,
        "actual": actual,
        "classes": datamodule.classes,
        "category_lengths": category_lengths,
        "train_batch": batch_shapes(train_batch),
        "validation_batch": batch_shapes(validation_batch),
        "cache": str(args.cache.resolve()),
        "cache_files": cache_files,
        "cache_bytes": sum(item["size"] for item in cache_files),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": sha256(args.config),
        "membership": membership,
        "membership_sha256": sha256(args.membership),
        "ntv3": str(args.ntv3.resolve()),
        "ntv3_sha256": sha256(args.ntv3),
        "registry_manifest": json.loads(args.registry_manifest.read_text()),
        "effective_data": data,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("PRIMARY_FILTERED_CACHE_PASS " + json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
