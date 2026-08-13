#!/usr/bin/env python3
"""Run scPRINT-2 small-v2 OpenProblems evaluation with raw full-no-assay embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from op_scib import (
    compute_op_scib_metrics,
    load_op_solution,
    prepare_op_scib_environment,
    save_op_scib_result,
)
from scprint2 import scPRINT2
from scprint2.tasks import Embedder, FinetuneBatchClass
from scprint2.tasks.cell_emb import compute_classification
from scprint2.utils import zero_shot_annotation_with_refinement
from scripts.op_classification_output import save_classification_output

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
ASSAY = "assay_ontology_term_id"
CELL_TYPE = "cell_type_ontology_term_id"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(checkpoint: str) -> scPRINT2:
    model = scPRINT2.load_from_checkpoint(
        checkpoint,
        precpt_gene_emb=None,
        gene_pos_file=None,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    model.mask_zeros = True
    return model


def embed_all(model: scPRINT2, adata, num_workers: int):
    embedded, _ = Embedder(
        how="random expr",
        max_len=3_200,
        num_workers=num_workers,
        pred_embedding=["all"],
        keep_all_labels_pred=True,
        doplot=False,
    )(model, adata)
    selected = [
        f"scprint_emb_{token}"
        for token in ["other", *model.classes]
        if token != ASSAY
    ]
    missing = [key for key in selected if key not in embedded.obsm]
    if missing:
        raise KeyError(f"Missing embedding blocks: {missing}")
    embedded.obsm["scprint_emb"] = np.concatenate(
        [np.asarray(embedded.obsm[key], dtype=np.float32) for key in selected],
        axis=1,
    )
    return embedded, selected


def classification_scores(embedded, model: scPRINT2, test_donors: list[str]) -> dict:
    class_columns = embedded.obs.columns[
        embedded.obs.columns.str.startswith("CL:")
    ]
    if len(class_columns) == 0:
        raise ValueError("No cell-type classification logits were returned")
    logits = embedded.obs.loc[:, class_columns].copy()
    embedded.obsm["classification_logits"] = logits.to_numpy(dtype=np.float32)
    embedded.obsm["classification_embedding"] = np.asarray(
        embedded.obsm["scprint_emb"], dtype=np.float32
    ).copy()
    embedded.uns["classification_logit_labels"] = class_columns.to_list()
    direct_labels = class_columns[logits.values.argmax(1)].values
    embedded.obs[f"pred_{CELL_TYPE}"] = direct_labels
    embedded.obs[f"pred_{CELL_TYPE}_direct"] = direct_labels

    def compute(view):
        return compute_classification(
            view,
            [CELL_TYPE],
            label_decoders=model.label_decoders,
            labels_hierarchy=model.labels_hierarchy,
        )[CELL_TYPE]

    held_out = embedded.obs["donor_id"].isin(test_donors)
    embedded.obs["classification_held_out"] = held_out
    scores = {
        "all_cells_direct": compute(embedded),
        "held_out_direct": compute(embedded[held_out]),
    }

    refined = zero_shot_annotation_with_refinement(
        logits.values, embedded, return_raw=True
    ).astype(np.float32)
    refined_logits = pd.DataFrame(
        refined,
        index=logits.index,
        columns=logits.columns,
        dtype=np.float32,
    )
    embedded.obsm["classification_refined_logits"] = refined
    smooth_labels = class_columns[
        zero_shot_annotation_with_refinement(refined, embedded)
    ].values
    embedded.obs[f"pred_{CELL_TYPE}"] = smooth_labels
    embedded.obs[f"pred_{CELL_TYPE}_smooth"] = smooth_labels
    scores["held_out_smooth"] = compute(embedded[held_out])

    if "seurat_clusters" in embedded.obs:
        embedded.obs["leiden"] = embedded.obs["seurat_clusters"].astype(str)
    if "leiden" not in embedded.obs:
        sc.pp.neighbors(embedded, use_rep="scprint_emb")
        sc.tl.leiden(embedded, resolution=4.0)
    for cluster in embedded.obs["leiden"].unique():
        in_cluster = embedded.obs["leiden"] == cluster
        winner = refined_logits.loc[in_cluster].values.sum(0).argmax()
        embedded.obs.loc[in_cluster, f"pred_{CELL_TYPE}"] = class_columns[winner]
    embedded.obs[f"pred_{CELL_TYPE}_cluster"] = embedded.obs[
        f"pred_{CELL_TYPE}"
    ].copy()
    scores["held_out_cluster"] = compute(embedded[held_out])
    return scores


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    prepare_op_scib_environment()
    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(args.input)
    model = load_model(args.checkpoint)
    if args.mode == "finetune":
        train_mask = ~adata.obs["donor_id"].isin(spec["test_donors"])
        model = FinetuneBatchClass(
            batch_key="donor_id",
            max_len=3_200,
            predict_keys=[
                CELL_TYPE,
                "disease_ontology_term_id",
                ASSAY,
                "self_reported_ethnicity_ontology_term_id",
                "sex_ontology_term_id",
            ],
            do_mmd_on=CELL_TYPE,
            batch_size=32,
            num_epochs=2,
            lr=0.0001,
            loss_scalers={CELL_TYPE: 6.0, "kl": 0},
        )(model=model, train_data=adata[train_mask])

    embedded, selected = embed_all(model, adata, args.num_workers)
    classification = classification_scores(
        embedded, model, spec["test_donors"]
    )
    if args.mode == "finetune":
        embedded.obsm["scprint_emb"] = np.asarray(
            embedded.obsm[f"scprint_emb_{CELL_TYPE}"], dtype=np.float32
        )
        embedding_name = "cell_type"
        embedding_blocks = [f"scprint_emb_{CELL_TYPE}"]
    else:
        embedding_name = "full_no_assay"
        embedding_blocks = selected
    method = f"scPRINT-2 small-v2 {args.mode} {embedding_name} raw"
    stem = f"{args.dataset}_scprint2_small_v2_{args.mode}_{embedding_name}"
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classification_only": args.classification_only,
        "dataset": spec["name"],
        "embedding": embedding_name,
        "embedding_blocks": embedding_blocks,
        "embedding_width": int(embedded.obsm["scprint_emb"].shape[1]),
        "mode": args.mode,
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

    scib = compute_op_scib_metrics(
        embedded,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=load_op_solution(spec["name"]),
        method_id=method,
    )
    save_op_scib_result(scib, output_dir / f"{stem}_op_scib.csv")
    print(scib.to_string(), flush=True)
    print(json.dumps(classification, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--mode", required=True, choices=("zeroshot", "finetune"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--classification-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
