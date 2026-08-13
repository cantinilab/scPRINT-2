#!/usr/bin/env python3
"""Run the scPRINT-1 DKD token-embedding aggregation ablation.

The embedding stage performs one forward pass retaining every cell token, then
builds PCA-50 representations from the same predictions.  The scoring stage
evaluates one cached representation with the repository's OpenProblems/scIB
implementation, which makes it suitable for a dependent Slurm array.
"""

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

DATASET_NAME = "cellxgene_census/dkd"
ASSAY_TOKEN = "assay_ontology_term_id"
CELL_TYPE_TOKEN = "cell_type_ontology_term_id"
TOKEN_VARIANT_NAMES = {
    "other": "other",
    CELL_TYPE_TOKEN: "cell_type",
    "disease_ontology_term_id": "disease",
    ASSAY_TOKEN: "assay",
    "self_reported_ethnicity_ontology_term_id": "ethnicity",
    "sex_ontology_term_id": "sex",
    "organism_ontology_term_id": "organism",
}
VARIANTS = (
    "cell_type",
    "mean_no_assay",
    "concat_no_assay",
    "mean_all",
    "concat_all",
)


def _pca50(values: np.ndarray) -> np.ndarray:
    n_comps = min(50, values.shape[0] - 1, values.shape[1])
    return np.asarray(
        sc.pp.pca(
            np.asarray(values, dtype=np.float32),
            n_comps=n_comps,
            chunked=True,
            chunk_size=2_000,
        ),
        dtype=np.float32,
    )


def _write_variant(
    output_dir: Path,
    name: str,
    obs,
    values: np.ndarray,
    token_names: list[str],
) -> None:
    result = ad.AnnData(obs=obs.copy())
    result.obsm["scprint_emb"] = _pca50(values)
    result.uns["ablation_variant"] = name
    result.uns["source_tokens"] = token_names
    result.write_h5ad(output_dir / f"dkd_scprint1_{name}_pca50.h5ad")


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
    )
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if not model.genes:
        raise RuntimeError("The checkpoint contains no model genes")

    source = sc.read_h5ad(args.input)
    runner = Embedder(
        how="random expr",
        max_len=2_300,
        num_workers=args.num_workers,
        pred_embedding=["all"],
        doclass=False,
        doplot=False,
    )
    embedded, _ = runner(model, source)

    expected_tokens = ["other", *model.classes]
    blocks: dict[str, np.ndarray] = {}
    for token in expected_tokens:
        key = f"scprint_emb_{token}"
        if key not in embedded.obsm:
            raise KeyError(f"Embedder did not return {key!r}")
        blocks[token] = np.asarray(embedded.obsm[key], dtype=np.float32)

    if CELL_TYPE_TOKEN not in blocks:
        raise KeyError(f"Checkpoint has no {CELL_TYPE_TOKEN!r} embedding")
    if ASSAY_TOKEN not in blocks:
        raise KeyError(f"Checkpoint has no {ASSAY_TOKEN!r} embedding")

    all_tokens = list(blocks)
    no_assay_tokens = [token for token in all_tokens if token != ASSAY_TOKEN]
    obs = embedded.obs[["donor_id", "cell_type"]]
    matrices = {
        "cell_type": blocks[CELL_TYPE_TOKEN],
        "mean_no_assay": np.mean(
            np.stack([blocks[token] for token in no_assay_tokens], axis=0), axis=0
        ),
        "concat_no_assay": np.concatenate(
            [blocks[token] for token in no_assay_tokens], axis=1
        ),
        "mean_all": np.mean(
            np.stack([blocks[token] for token in all_tokens], axis=0), axis=0
        ),
        "concat_all": np.concatenate(
            [blocks[token] for token in all_tokens], axis=1
        ),
    }
    variant_tokens = {
        "cell_type": [CELL_TYPE_TOKEN],
        "mean_no_assay": no_assay_tokens,
        "concat_no_assay": no_assay_tokens,
        "mean_all": all_tokens,
        "concat_all": all_tokens,
    }
    if args.token_ablation:
        for token, variant_name in TOKEN_VARIANT_NAMES.items():
            if token not in blocks:
                raise KeyError(f"Checkpoint has no {token!r} embedding")
            matrices[variant_name] = blocks[token]
            variant_tokens[variant_name] = [token]
        no_other_tokens = [token for token in all_tokens if token != "other"]
        matrices["mean_no_other"] = np.mean(
            np.stack([blocks[token] for token in no_other_tokens], axis=0), axis=0
        )
        variant_tokens["mean_no_other"] = no_other_tokens

        raw = ad.AnnData(obs=obs.copy())
        for token, values in blocks.items():
            raw.obsm[f"scprint_emb_{token}"] = values
        raw.uns["source_tokens"] = all_tokens
        raw.write_h5ad(
            output_dir / "dkd_scprint1_all_token_embeddings.h5ad",
            compression="gzip",
        )

    variants = list(matrices)
    for name in variants:
        print(f"Writing {name}: input shape={matrices[name].shape}", flush=True)
        _write_variant(output_dir, name, obs, matrices[name], variant_tokens[name])

    manifest = {
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "all_tokens": all_tokens,
        "no_assay_tokens": no_assay_tokens,
        "variants": variants,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def score(args: argparse.Namespace) -> None:
    prepare_op_scib_environment()
    output_dir = Path(args.output_dir)
    embedding_path = output_dir / f"dkd_scprint1_{args.variant}_pca50.h5ad"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Unknown or ungenerated variant: {embedding_path}")
    integrated = sc.read_h5ad(embedding_path)
    source = sc.read_h5ad(args.input, backed="r")
    if integrated.n_obs != source.n_obs or not integrated.obs_names.equals(
        source.obs_names
    ):
        raise ValueError("Cached embedding rows do not match the processed DKD input")
    # Keep the technical metadata and library-size fingerprint used by
    # op_scib._align_solution to validate positional alignment against the
    # reconstructed OpenProblems reference, whose observation IDs differ.
    integrated.obs = source.obs.copy()
    source.file.close()
    solution = load_op_solution(DATASET_NAME)
    result = compute_op_scib_metrics(
        integrated,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=solution,
        method_id=f"scPRINT-1 {args.variant} PCA50",
    )
    path = output_dir / f"dkd_scprint1_{args.variant}_op_scib.csv"
    save_op_scib_result(result, path)
    print(result.to_string(), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed")
    embed_parser.add_argument("--input", required=True)
    embed_parser.add_argument("--checkpoint", required=True)
    embed_parser.add_argument("--output-dir", required=True)
    embed_parser.add_argument("--num-workers", type=int, default=8)
    embed_parser.add_argument("--seed", type=int, default=0)
    embed_parser.add_argument("--token-ablation", action="store_true")

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--output-dir", required=True)
    score_parser.add_argument("--variant", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.command == "embed":
        embed(parsed)
    else:
        score(parsed)
