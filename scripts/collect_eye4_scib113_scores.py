#!/usr/bin/env python3
"""Collect independently scored eye4 scIB 1.1.3 method artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TOTAL_FORMULA = "0.4 * Batch correction + 0.6 * Bio conservation"
INVARIANT_METADATA_KEYS = (
    "versions",
    "source_sha256",
    "batch_key",
    "label_key",
    "cells",
    "labels",
    "batches",
    "expression_connectivities_sha256",
    "total_formula",
)


def parse_input_spec(spec: str) -> tuple[str, Path]:
    """Parse a ``METHOD=CSV`` command-line input specification."""
    method, separator, raw_path = spec.partition("=")
    if not separator or not method.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input must be METHOD=CSV")
    return method.strip(), Path(raw_path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_unique(values: list[str], description: str) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise RuntimeError(f"Duplicate {description}: {duplicates}")


def _read_one_method(
    method: str, score_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    metadata_path = score_path.with_suffix(".metadata.json")
    complete_path = score_path.with_suffix(".COMPLETE")
    for path in (score_path, metadata_path, complete_path):
        if not path.is_file():
            raise RuntimeError(f"Incomplete score artifact for {method}: missing {path}")
    if complete_path.read_text() != "COMPLETE\n":
        raise RuntimeError(f"Invalid completion marker for {method}: {complete_path}")

    table = pd.read_csv(score_path)
    if len(table) != 1:
        raise RuntimeError(
            f"Expected one score row for {method} in {score_path}, found {len(table)}"
        )
    row = table.iloc[0].to_dict()
    if row.get("Method") != method:
        raise RuntimeError(
            f"Method mismatch for {score_path}: expected {method!r}, "
            f"found {row.get('Method')!r}"
        )

    metadata = json.loads(metadata_path.read_text())
    embeddings = metadata.get("embeddings")
    if not isinstance(embeddings, dict) or set(embeddings) != {method}:
        raise RuntimeError(
            f"Expected metadata['embeddings'] to contain only {method!r}"
        )
    return row, metadata, metadata_path, complete_path


def _validate_total(table: pd.DataFrame, metadata: list[dict[str, Any]]) -> None:
    required = {"Batch correction", "Bio conservation", "Total"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"Missing score columns required for Total: {missing}")
    expected = 0.4 * table["Batch correction"] + 0.6 * table["Bio conservation"]
    if not np.allclose(table["Total"], expected, rtol=0, atol=1e-12):
        raise RuntimeError("Total score formula validation failed")
    if any(item.get("total_formula") != TOTAL_FORMULA for item in metadata):
        raise RuntimeError(f"Expected total_formula={TOTAL_FORMULA!r}")


def _validate_metadata(metadata: list[dict[str, Any]]) -> None:
    reference = metadata[0]
    missing = [key for key in INVARIANT_METADATA_KEYS if key not in reference]
    if missing:
        raise RuntimeError(f"Reference metadata is missing invariant keys: {missing}")
    for item in metadata[1:]:
        for key in INVARIANT_METADATA_KEYS:
            if item.get(key) != reference[key]:
                raise RuntimeError(f"Inconsistent score metadata for {key}")
    if reference["versions"].get("scib") != "1.1.3":
        raise RuntimeError("Score collection requires scib==1.1.3")


def _validated_graph(method: str, metadata: dict[str, Any]) -> tuple[Path, dict]:
    provenance = dict(metadata["embeddings"][method])
    before = provenance.get("connectivities_sha256_before_scib")
    after = provenance.get("connectivities_sha256_after_scib")
    if not before or before != after:
        raise RuntimeError(f"scIB changed or did not hash the KNN graph for {method}")
    try:
        graph_path = Path(provenance["precomputed_graph"])
        declared_hash = provenance["precomputed_graph_sha256"]
    except KeyError as error:
        raise RuntimeError(f"Missing precomputed graph provenance for {method}") from error
    if not graph_path.is_file():
        raise RuntimeError(f"Missing precomputed graph for {method}: {graph_path}")
    actual_hash = sha256(graph_path)
    if actual_hash != declared_hash:
        raise RuntimeError(f"Precomputed graph hash mismatch for {method}")
    return graph_path, provenance


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)


def collect_scores(
    inputs: list[tuple[str, Path]], expected_methods: list[str], output: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and combine independent one-method scIB 1.1.3 results."""
    if not inputs:
        raise RuntimeError("At least one --input is required")
    if not expected_methods:
        raise RuntimeError("At least one --expected-method is required")
    input_methods = [method for method, _ in inputs]
    _require_unique(input_methods, "input methods")
    _require_unique(expected_methods, "expected methods")
    if set(input_methods) != set(expected_methods):
        missing = sorted(set(expected_methods) - set(input_methods))
        extra = sorted(set(input_methods) - set(expected_methods))
        raise RuntimeError(f"Method mismatch; missing={missing}, extra={extra}")

    metadata_output = output.with_suffix(".metadata.json")
    complete_output = output.with_suffix(".COMPLETE")
    for path in (output, metadata_output, complete_output):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    rows: list[dict[str, Any]] = []
    metadata_items: list[dict[str, Any]] = []
    artifact_records: list[dict[str, str]] = []
    graph_sources: dict[str, tuple[Path, dict[str, Any]]] = {}
    by_method = dict(inputs)
    for method in expected_methods:
        score_path = by_method[method]
        row, metadata, metadata_path, complete_path = _read_one_method(
            method, score_path
        )
        rows.append(row)
        metadata_items.append(metadata)
        graph_sources[method] = _validated_graph(method, metadata)
        artifact_records.append(
            {
                "method": method,
                "score": str(score_path.resolve()),
                "score_sha256": sha256(score_path),
                "metadata": str(metadata_path.resolve()),
                "metadata_sha256": sha256(metadata_path),
                "complete": str(complete_path.resolve()),
            }
        )

    table = pd.DataFrame(rows)
    _validate_metadata(metadata_items)
    _validate_total(table, metadata_items)
    table = table.sort_values("Total", ascending=False, kind="mergesort").reset_index(
        drop=True
    )

    graph_dir = output.parent / "precomputed_graphs"
    destinations: dict[str, Path] = {}
    destination_names: list[str] = []
    for method, (source, _) in graph_sources.items():
        destination = graph_dir / source.name
        destinations[method] = destination
        destination_names.append(destination.name)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
    _require_unique(destination_names, "precomputed graph filenames")

    output.parent.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    merged_embeddings: dict[str, dict[str, Any]] = {}
    for method in expected_methods:
        source, provenance = graph_sources[method]
        destination = destinations[method]
        _copy_exclusive(source, destination)
        copied_hash = sha256(destination)
        if copied_hash != provenance["precomputed_graph_sha256"]:
            raise RuntimeError(f"Copied precomputed graph hash mismatch for {method}")
        provenance["precomputed_graph"] = str(destination.resolve())
        provenance["precomputed_graph_sha256"] = copied_hash
        merged_embeddings[method] = provenance

    combined_metadata = dict(metadata_items[0])
    combined_metadata["embeddings"] = merged_embeddings
    combined_metadata["method_order"] = expected_methods
    combined_metadata["validated_individual_artifacts"] = artifact_records

    with output.open("x") as handle:
        table.to_csv(handle, index=False)
    with metadata_output.open("x") as handle:
        json.dump(combined_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with complete_output.open("x") as handle:
        handle.write("COMPLETE\n")
    return table, combined_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", action="append", required=True, type=parse_input_spec
    )
    parser.add_argument("--expected-method", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table, _ = collect_scores(args.input, args.expected_method, args.output)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
