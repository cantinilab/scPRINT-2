#!/usr/bin/env python3
"""Recompute the four OpenProblems datasets with the corrected scPRINT-1 gene map."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
from scprint2.tasks.cell_emb import compute_classification
from scripts.op_classification_output import save_classification_output

CELL_TYPE = "cell_type_ontology_term_id"
DATASETS = {
    "dkd": {
        "name": "cellxgene_census/dkd",
        "test_donors": ["control_3"],
    },
    "gtex_v9": {
        "name": "cellxgene_census/gtex_v9",
        "test_donors": ["GTEX-16BQI"],
    },
    "hypomap": {
        "name": "cellxgene_census/hypomap",
        "test_donors": ["SRR9000488"],
    },
    "mouse_pancreas_atlas": {
        "name": "cellxgene_census/mouse_pancreas_atlas",
        "test_donors": [
            "mouse_pancreatic_islet_atlas_Hrovatin__VSG__MUC13639"
        ],
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_scores(adata, model, test_donors: list[str]) -> dict:
    def compute(view):
        return compute_classification(
            view,
            [CELL_TYPE],
            label_decoders=model.label_decoders,
            labels_hierarchy=model.labels_hierarchy,
        )[CELL_TYPE]

    held_out = adata.obs["donor_id"].isin(test_donors)
    if not held_out.any():
        raise ValueError(f"No cells found for held-out donors {test_donors}")
    adata.obs["classification_held_out"] = held_out
    prediction_column = f"pred_{CELL_TYPE}"
    adata.obs[f"{prediction_column}_direct"] = adata.obs[
        prediction_column
    ].copy()
    return {
        "all_cells_direct": compute(adata),
        "held_out_direct": compute(adata[held_out]),
        "held_out_cells": int(held_out.sum()),
    }


def pca50(values: np.ndarray) -> np.ndarray:
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


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    prepare_op_scib_environment()
    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    model = scPRINT2.load_from_checkpoint(
        args.checkpoint,
        precpt_gene_emb=None,
        gene_pos_file=None,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    embedded, _ = Embedder(
        how="random expr",
        max_len=2_300,
        num_workers=args.num_workers,
        pred_embedding=[CELL_TYPE],
        doplot=False,
    )(model, adata)
    classification = classification_scores(
        embedded, model, spec["test_donors"]
    )
    embedded.obsm["scprint_emb_cell_type_raw"] = np.asarray(
        embedded.obsm["scprint_emb"], dtype=np.float32
    ).copy()
    embedded.obsm["scprint_emb"] = pca50(embedded.obsm["scprint_emb"])

    stem = f"{args.dataset}_scprint1_corrected_cell_type_pca50"
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classification_only": args.classification_only,
        "dataset": spec["name"],
        "embedding": "cell_type PCA50",
        "embedding_width": int(embedded.obsm["scprint_emb"].shape[1]),
        "seed": args.seed,
        "test_donors": spec["test_donors"],
    }
    save_classification_output(
        embedded,
        output_dir / f"{stem}_classification_output.h5ad",
        metadata={
            "label_decoders": model.label_decoders,
            "labels_hierarchy": model.labels_hierarchy,
            **manifest,
        },
    )
    (output_dir / f"{stem}_classification.json").write_text(
        json.dumps(classification, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / f"{stem}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.classification_only:
        print(json.dumps(classification, indent=2, sort_keys=True), flush=True)
        return

    result = compute_op_scib_metrics(
        embedded,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=load_op_solution(spec["name"]),
        method_id="scPRINT-1 corrected cell_type PCA50",
    )
    save_op_scib_result(result, output_dir / f"{stem}_op_scib.csv")
    print(result.to_string(), flush=True)
    print(json.dumps(classification, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
