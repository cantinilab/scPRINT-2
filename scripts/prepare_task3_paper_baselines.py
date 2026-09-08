#!/usr/bin/env python3
"""Build deterministic PCA and random task3 baselines for the paper evaluator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=50)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    source = sc.read_h5ad(args.source)
    sample = source.X[: min(128, source.n_obs)]
    values = sample.data if hasattr(sample, "data") else np.asarray(sample).ravel()
    normalized_from_counts = bool(
        values.size
        and np.all(values >= 0)
        and np.allclose(values, np.rint(values))
    )
    if normalized_from_counts:
        sc.pp.normalize_total(source, target_sum=10_000)
        sc.pp.log1p(source)
    sc.pp.pca(
        source,
        n_comps=args.n_components,
        svd_solver="arpack",
        use_highly_variable=False,
    )

    pca = np.asarray(source.obsm["X_pca"], dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    random_embedding = rng.random(pca.shape, dtype=np.float32)

    result = ad.AnnData(
        X=np.empty((source.n_obs, 0), dtype=np.float32),
        obs=pd.DataFrame(index=source.obs_names.copy()),
    )
    result.obsm["X_pca"] = pca
    result.obsm["random"] = random_embedding
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output, compression="lzf")

    metadata = {
        "protocol": "task3 paper-compatible raw PCA and random baselines",
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "seed": args.seed,
        "n_components": args.n_components,
        "normalized_from_counts": normalized_from_counts,
        "preprocessing": [
            "scanpy.pp.normalize_total(target_sum=10000)",
            "scanpy.pp.log1p",
            "scanpy.pp.pca(n_comps=50, svd_solver=arpack, use_highly_variable=False)",
        ],
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("scanpy", "anndata", "numpy")
        },
        "n_obs": int(source.n_obs),
        "pca_shape": list(pca.shape),
        "random_shape": list(random_embedding.shape),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
