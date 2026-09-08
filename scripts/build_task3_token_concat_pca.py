#!/usr/bin/env python3
"""Concatenate selected scPRINT token embeddings and reduce them to PCA-50."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import scanpy as sc

from scprint2.evaluation.cross_species import (
    TOKEN_KEYS,
    all_token_keys_except_assay_and_organism,
    all_token_keys_except_organism,
    build_token_concat_pca,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--token-policy",
        choices=(
            "selected-four",
            "all-except-assay-organism",
            "all-except-organism",
        ),
        default="selected-four",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = sc.read_h5ad(args.input, backed="r")
    if args.token_policy == "all-except-assay-organism":
        token_keys = all_token_keys_except_assay_and_organism(source)
    elif args.token_policy == "all-except-organism":
        token_keys = all_token_keys_except_organism(source)
    else:
        token_keys = TOKEN_KEYS
    embedding, metadata = build_token_concat_pca(
        source, seed=args.seed, token_keys=token_keys
    )

    output = ad.AnnData(obs=source.obs.copy())
    output.obsm["token_concat_pca50"] = embedding
    output.uns["token_concat_pca50"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(args.output, compression="lzf")
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "input": str(args.input.resolve()),
                "token_policy": args.token_policy,
                **output.uns["token_concat_pca50"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
