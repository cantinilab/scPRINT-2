#!/usr/bin/env python3
"""Reconstruct and validate the OpenProblems no-integration baseline.

Example on Jean Zay (download the common dataset on the submit node first):

    python scripts/validate_op_no_integration.py \
      --input-common "$WORK/openproblems_common/cellxgene_census/dkd/log_cp10k/dataset.h5ad" \
      --dataset dkd \
      --output-root "$WORK/openproblems_reconstructed"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad

from op_scib import (
    OP_NO_INTEGRATION_EXPECTED,
    compare_op_scores,
    compute_op_scib_metrics,
    make_op_no_integration,
    reconstruct_op_batch_reference,
    save_op_scib_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-common", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--silhouette-backend", choices=("jax", "scib"), default="jax"
    )
    parser.add_argument("--silhouette-chunk-size", type=int, default=1024)
    parser.add_argument("--skip-kbet", action="store_true")
    parser.add_argument("--skip-write-reference", action="store_true")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="write the reconstructed reference without benchmarking no_integration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading common dataset: {args.input_common}", flush=True)
    common = ad.read_h5ad(args.input_common)
    print(common, flush=True)

    print("Reconstructing OpenProblems normalization and task PCA...", flush=True)
    reference, normalization_check = reconstruct_op_batch_reference(common)
    del common
    (output_dir / "normalization_check.json").write_text(
        json.dumps(normalization_check, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(normalization_check, indent=2, sort_keys=True), flush=True)

    if not args.skip_write_reference:
        reference_path = output_dir / "output_solution.h5ad"
        print(f"Writing reconstructed reference: {reference_path}", flush=True)
        reference.write_h5ad(reference_path, compression="gzip")

    if args.reference_only:
        print("Reference-only mode: skipping no_integration metrics.", flush=True)
        return

    integrated = make_op_no_integration(reference)
    print("Computing scIB metrics for no_integration...", flush=True)
    scores = compute_op_scib_metrics(
        integrated,
        embedding_key="X_emb",
        batch_key="batch",
        label_key="cell_type",
        solution=reference,
        method_id="no_integration",
        silhouette_backend=args.silhouette_backend,
        silhouette_chunk_size=args.silhouette_chunk_size,
        compute_kbet=not args.skip_kbet,
        strict=False,
        verbose=True,
    )
    result_path = save_op_scib_result(
        scores, output_dir / "no_integration_op_scib.csv"
    )
    print(scores.T, flush=True)

    expected = OP_NO_INTEGRATION_EXPECTED.get(args.dataset)
    if expected is not None:
        comparison = compare_op_scores(scores, expected)
        comparison_path = output_dir / "no_integration_comparison.csv"
        comparison.to_csv(comparison_path)
        print("\nComparison with published OpenProblems scores:", flush=True)
        print(comparison, flush=True)
    else:
        comparison_path = None
        print(f"No published validation row configured for {args.dataset!r}.")

    print(f"Metric output: {result_path}", flush=True)
    if comparison_path is not None:
        print(f"Comparison output: {comparison_path}", flush=True)


if __name__ == "__main__":
    main()
