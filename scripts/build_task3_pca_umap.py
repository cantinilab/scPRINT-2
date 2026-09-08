#!/usr/bin/env python3
"""Build a Task3 PCA UMAP with the same graph protocol as the scored methods."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def obs_names_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-baseline", required=True, type=Path)
    parser.add_argument("--reference-coordinates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--neighbors-random-state", type=int, default=0)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--umap-random-state", type=int, default=42)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    baseline = ad.read_h5ad(args.pca_baseline, backed="r")
    reference = ad.read_h5ad(args.reference_coordinates, backed="r")
    if not baseline.obs_names.equals(reference.obs_names):
        raise ValueError("PCA baseline and scored-coordinate cell order differ")
    if "X_pca" not in baseline.obsm:
        raise KeyError(f"X_pca missing from {args.pca_baseline}")

    coordinates = np.asarray(baseline.obsm["X_pca"], dtype=np.float32)
    result = ad.AnnData(
        X=sparse.csr_matrix((reference.n_obs, 0), dtype=np.float32),
        obs=reference.obs.copy(),
    )
    result.obsm["X_embedding"] = coordinates
    sc.pp.neighbors(
        result,
        n_neighbors=args.n_neighbors,
        use_rep="X_embedding",
        metric=args.metric,
        random_state=args.neighbors_random_state,
    )
    sc.tl.umap(
        result,
        min_dist=args.min_dist,
        random_state=args.umap_random_state,
    )

    compact = ad.AnnData(
        X=sparse.csr_matrix((reference.n_obs, 0), dtype=np.float32),
        obs=reference.obs.copy(),
    )
    compact.obsm["X_umap_pca"] = np.asarray(result.obsm["X_umap"], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compact.write_h5ad(args.output)

    metadata = {
        "pca_baseline": str(args.pca_baseline.resolve()),
        "reference_coordinates": str(args.reference_coordinates.resolve()),
        "output": str(args.output.resolve()),
        "n_cells": compact.n_obs,
        "pca_shape": list(coordinates.shape),
        "obs_names_sha256": obs_names_sha256(compact.obs_names.astype(str).tolist()),
        "neighbors": {
            "n_neighbors": args.n_neighbors,
            "metric": args.metric,
            "use_rep": "X_pca from paper-compatible PCA-50 baseline",
            "random_state": args.neighbors_random_state,
        },
        "umap": {
            "min_dist": args.min_dist,
            "random_state": args.umap_random_state,
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("scanpy", "anndata", "numpy")
        },
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".COMPLETE").touch()


if __name__ == "__main__":
    main()
