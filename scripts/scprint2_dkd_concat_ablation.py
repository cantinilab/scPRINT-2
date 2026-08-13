#!/usr/bin/env python3
"""Compare raw scPRINT-2 token concatenations on DKD."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch

from op_scib import (
    compute_op_scib_metrics,
    load_op_solution,
    prepare_op_scib_environment,
    save_op_scib_result,
)
from scprint2 import scPRINT2
from scprint2.tasks import Embedder

DATASET = "cellxgene_census/dkd"
ASSAY = "assay_ontology_term_id"
CELL_TYPE = "cell_type_ontology_term_id"
RAW_VARIANTS = ("cell_type", "full", "full_no_assay")


def embed(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = scPRINT2.load_from_checkpoint(
        args.checkpoint,
        precpt_gene_emb=None,
        gene_pos_file=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    source = sc.read_h5ad(args.input)
    embedded, _ = Embedder(
        how="random expr",
        max_len=3_200,
        num_workers=args.num_workers,
        pred_embedding=["all"],
        doclass=False,
        doplot=False,
    )(model, source)

    tokens = ["other", *model.classes]
    blocks = {}
    for token in tokens:
        key = f"scprint_emb_{token}"
        if key not in embedded.obsm:
            raise KeyError(f"Embedder did not return {key!r}")
        blocks[token] = np.asarray(embedded.obsm[key], dtype=np.float32)
    if CELL_TYPE not in blocks or ASSAY not in blocks:
        raise KeyError("Required cell-type or assay token is absent")

    obs = embedded.obs[["donor_id", "cell_type"]]
    raw = ad.AnnData(obs=obs.copy())
    for token, values in blocks.items():
        raw.obsm[f"scprint_emb_{token}"] = values
    raw.write_h5ad(output_dir / "dkd_scprint2_all_token_embeddings.h5ad", compression="gzip")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "tokens": tokens,
                "widths": {token: int(value.shape[1]) for token, value in blocks.items()},
                "raw_variants": list(RAW_VARIANTS),
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def score(args: argparse.Namespace) -> None:
    if args.variant not in RAW_VARIANTS:
        raise ValueError(f"Unknown variant: {args.variant}")
    prepare_op_scib_environment()
    output_dir = Path(args.output_dir)
    integrated = sc.read_h5ad(output_dir / "dkd_scprint2_all_token_embeddings.h5ad")
    token_keys = list(integrated.obsm)
    assay_key = f"scprint_emb_{ASSAY}"
    cell_type_key = f"scprint_emb_{CELL_TYPE}"
    if args.variant == "cell_type":
        selected = [cell_type_key]
    elif args.variant == "full_no_assay":
        selected = [key for key in token_keys if key != assay_key]
    else:
        selected = token_keys
    integrated.obsm["scprint_emb"] = np.concatenate(
        [np.asarray(integrated.obsm[key], dtype=np.float32) for key in selected],
        axis=1,
    )
    source = sc.read_h5ad(args.input, backed="r")
    if integrated.n_obs != source.n_obs or not integrated.obs_names.equals(source.obs_names):
        raise ValueError("Cached embedding rows do not match the processed DKD input")
    integrated.obs = source.obs.copy()
    source.file.close()
    result = compute_op_scib_metrics(
        integrated,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=load_op_solution(DATASET),
        method_id=f"scPRINT-2 {args.variant} raw",
    )
    save_op_scib_result(
        result, output_dir / f"dkd_scprint2_{args.variant}_raw_op_scib.csv"
    )
    print(result.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    embedding = commands.add_parser("embed")
    embedding.add_argument("--input", required=True)
    embedding.add_argument("--checkpoint", required=True)
    embedding.add_argument("--output-dir", required=True)
    embedding.add_argument("--num-workers", type=int, default=8)
    embedding.add_argument("--seed", type=int, default=0)
    scoring = commands.add_parser("score")
    scoring.add_argument("--input", required=True)
    scoring.add_argument("--output-dir", required=True)
    scoring.add_argument("--variant", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    embed(args) if args.command == "embed" else score(args)
