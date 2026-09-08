#!/usr/bin/env python3
"""Build expression-PCA and deterministic random baselines for scIB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import scanpy as sc

from scprint2.evaluation.cross_species import build_expression_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    source = sc.read_h5ad(args.input)
    pca, random, metadata = build_expression_baselines(source, seed=args.seed)
    output = ad.AnnData(obs=source.obs.copy())
    output.obsm["X_pca"] = pca
    output.obsm["random"] = random
    output.uns["expression_scib_baselines"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output, compression="lzf")
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "cells": int(source.n_obs),
                "genes": int(source.n_vars),
                **metadata,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
