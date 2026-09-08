"""OpenProblems benchmark helpers for scPRINT-2 embeddings."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from scprint2 import scPRINT2
from scprint2.evaluation.classification_output import save_classification_output
from scprint2.evaluation.openproblems_scib import (
    compute_op_scib_metrics,
    load_op_solution,
    prepare_op_scib_environment,
    save_op_scib_result,
)
from scprint2.tasks import Embedder, FinetuneBatchClass
from scprint2.tasks.cell_emb import compute_classification
from scprint2.utils import zero_shot_annotation_with_refinement

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
        "test_donors": ["mouse_pancreatic_islet_atlas_Hrovatin__VSG__MUC13639"],
    },
}
ASSAY = "assay_ontology_term_id"
CELL_TYPE = "cell_type_ontology_term_id"


@dataclass(frozen=True)
class OpenProblemsConfig:
    """Python-facing configuration used by the explanatory notebooks."""

    dataset: str
    mode: str
    input: Path | str
    checkpoint: Path | str
    output_dir: Path | str
    embedding_view: str = "full_no_assay"
    num_workers: int = 8
    seed: int = 0
    mmd_mode: str = "fixed"
    mmd_weight: float = 0.03
    mmd_batch_key: str = "donor_id"
    classification_only: bool = False
    scib_only: bool = False

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown OpenProblems dataset: {self.dataset}")
        if self.mode not in {"zeroshot", "finetune"}:
            raise ValueError(f"Unsupported OpenProblems mode: {self.mode}")
        if self.embedding_view not in {"full_no_assay", "cell_type"}:
            raise ValueError(f"Unsupported embedding view: {self.embedding_view}")
        if self.mmd_mode not in {"fixed", "off"}:
            raise ValueError(f"Unsupported MMD mode: {self.mmd_mode}")
        if self.mmd_weight < 0:
            raise ValueError("mmd_weight must be non-negative")
        if not self.mmd_batch_key:
            raise ValueError("mmd_batch_key must name an adata.obs column")
        if self.classification_only and self.scib_only:
            raise ValueError("classification_only and scib_only are mutually exclusive")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("input", "checkpoint", "output_dir"):
            values[key] = str(values[key])
        return values

    def as_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))


@dataclass
class OpenProblemsResult:
    classification: dict[str, Any]
    scib: pd.DataFrame | None
    manifest: dict[str, Any]
    artifacts: dict[str, Path]

    def artifact_table(self) -> pd.Series:
        return pd.Series(
            {name: str(path) for name, path in self.artifacts.items()},
            name="artifacts",
        )


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


def load_openproblems_dataset(config: OpenProblemsConfig | argparse.Namespace):
    """Load one benchmark dataset and validate its held-out donor contract."""
    args = config.as_namespace() if isinstance(config, OpenProblemsConfig) else config
    adata = sc.read_h5ad(args.input)
    required = {"donor_id", CELL_TYPE}
    if args.mode == "finetune":
        required.add(args.mmd_batch_key)
    missing = sorted(required - set(adata.obs))
    if missing:
        raise KeyError(f"Missing OpenProblems obs columns: {missing}")
    test_donors = DATASETS[args.dataset]["test_donors"]
    if not adata.obs["donor_id"].isin(test_donors).any():
        raise ValueError(f"No cells found for held-out donors {test_donors}")
    return adata


def summarize_openproblems_dataset(
    adata, config: OpenProblemsConfig | argparse.Namespace
) -> pd.Series:
    """Expose training/test sizes and grouping keys for notebook review."""
    args = config.as_namespace() if isinstance(config, OpenProblemsConfig) else config
    test_donors = DATASETS[args.dataset]["test_donors"]
    held_out = adata.obs["donor_id"].isin(test_donors)
    return pd.Series(
        {
            "dataset": DATASETS[args.dataset]["name"],
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "training_cells": int((~held_out).sum()),
            "held_out_cells": int(held_out.sum()),
            "donors": int(adata.obs["donor_id"].nunique()),
            "cell_types": int(adata.obs[CELL_TYPE].astype(str).nunique()),
            "held_out_donors": ", ".join(test_donors),
            "mmd_batch_key": args.mmd_batch_key if args.mode == "finetune" else None,
        },
        name="OpenProblems dataset audit",
    )


def finetune_openproblems_model(
    model: scPRINT2,
    adata,
    config: OpenProblemsConfig | argparse.Namespace,
):
    """Fine-tune scPRINT-2 while keeping the held-out donor outside training."""
    args = config.as_namespace() if isinstance(config, OpenProblemsConfig) else config
    if args.mode != "finetune":
        return model
    test_donors = DATASETS[args.dataset]["test_donors"]
    train_mask = ~adata.obs["donor_id"].isin(test_donors)
    do_mmd_on = None if args.mmd_mode == "off" else CELL_TYPE
    return FinetuneBatchClass(
        batch_key=args.mmd_batch_key,
        max_len=3_200,
        predict_keys=[
            CELL_TYPE,
            "disease_ontology_term_id",
            ASSAY,
            "self_reported_ethnicity_ontology_term_id",
            "sex_ontology_term_id",
        ],
        do_mmd_on=do_mmd_on,
        batch_size=32,
        num_epochs=2,
        lr=0.0001,
        loss_scalers={CELL_TYPE: 6.0, "kl": 0, "mmd": args.mmd_weight},
    )(model=model, train_data=adata[train_mask])


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
        f"scprint_emb_{token}" for token in ["other", *model.classes] if token != ASSAY
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
    class_columns = embedded.obs.columns[embedded.obs.columns.str.startswith("CL:")]
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
    if not held_out.any():
        raise ValueError(f"No cells found for held-out donors {test_donors}")
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
    embedded.obs[f"pred_{CELL_TYPE}_cluster"] = embedded.obs[f"pred_{CELL_TYPE}"].copy()
    scores["held_out_cluster"] = compute(embedded[held_out])
    return scores


def select_embedding_view(
    embedded,
    selected: list[str],
    *,
    mode: str,
    embedding_view: str,
) -> tuple[str, list[str]]:
    if mode == "finetune" or embedding_view == "cell_type":
        key = f"scprint_emb_{CELL_TYPE}"
        embedded.obsm["scprint_emb"] = np.asarray(embedded.obsm[key], dtype=np.float32)
        return "cell_type", [key]
    return "full_no_assay", selected


def build_manifest(
    args: argparse.Namespace,
    *,
    spec: dict[str, Any],
    embedded,
    embedding_name: str,
    embedding_blocks: list[str],
) -> dict[str, Any]:
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classification_only": args.classification_only,
        "scib_only": args.scib_only,
        "dataset": spec["name"],
        "embedding": embedding_name,
        "embedding_blocks": embedding_blocks,
        "embedding_width": int(embedded.obsm["scprint_emb"].shape[1]),
        "mode": args.mode,
        "mmd_mode": args.mmd_mode if args.mode == "finetune" else "not_applicable",
        "mmd_weight": (
            args.mmd_weight
            if args.mode == "finetune" and args.mmd_mode == "fixed"
            else 0.0
        ),
        "mmd_batch_key": args.mmd_batch_key if args.mode == "finetune" else None,
        "seed": args.seed,
        "test_donors": spec["test_donors"],
    }


def run_openproblems_benchmark(
    config: OpenProblemsConfig | argparse.Namespace,
) -> OpenProblemsResult:
    args = config.as_namespace() if isinstance(config, OpenProblemsConfig) else config
    if args.classification_only and args.scib_only:
        raise ValueError("classification_only and scib_only are mutually exclusive")
    set_seed(args.seed)
    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adata = load_openproblems_dataset(args)
    model = load_model(args.checkpoint)
    model = finetune_openproblems_model(model, adata, args)

    embedded, selected = embed_all(model, adata, args.num_workers)
    classification = (
        {}
        if args.scib_only
        else classification_scores(embedded, model, spec["test_donors"])
    )
    embedding_name, embedding_blocks = select_embedding_view(
        embedded,
        selected,
        mode=args.mode,
        embedding_view=args.embedding_view,
    )
    method = f"scPRINT-2 small-v2 {args.mode} {embedding_name} raw"
    stem = f"{args.dataset}_scprint2_small_v2_{args.mode}_{embedding_name}"
    manifest = build_manifest(
        args,
        spec=spec,
        embedded=embedded,
        embedding_name=embedding_name,
        embedding_blocks=embedding_blocks,
    )
    artifacts: dict[str, Path] = {}
    if not args.scib_only:
        classification_output = output_dir / f"{stem}_classification_output.h5ad"
        save_classification_output(
            embedded,
            classification_output,
            metadata={
                "label_decoders": model.label_decoders,
                "labels_hierarchy": model.labels_hierarchy,
                **manifest,
            },
        )
        classification_path = output_dir / f"{stem}_classification.json"
        classification_path.write_text(
            json.dumps(classification, indent=2, sort_keys=True) + "\n"
        )
        artifacts["classification_output"] = classification_output
        artifacts["classification_scores"] = classification_path
    else:
        # Preserve the expensive inference before running memory-heavy scIB
        # metrics, so a late evaluator failure can be resumed post hoc.
        scib_embedding = output_dir / f"{stem}_scib_embedding.h5ad"
        save_classification_output(
            embedded,
            scib_embedding,
            metadata={
                "label_decoders": model.label_decoders,
                "labels_hierarchy": model.labels_hierarchy,
                **manifest,
            },
        )
        artifacts["scib_embedding"] = scib_embedding
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifacts["manifest"] = manifest_path
    if args.classification_only:
        print(json.dumps(classification, indent=2, sort_keys=True), flush=True)
        return OpenProblemsResult(classification, None, manifest, artifacts)

    prepare_op_scib_environment()
    scib = compute_op_scib_metrics(
        embedded,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=load_op_solution(spec["name"]),
        method_id=method,
    )
    scib_path = save_op_scib_result(scib, output_dir / f"{stem}_op_scib.csv")
    artifacts["scib_scores"] = scib_path
    print(scib.to_string(), flush=True)
    print(json.dumps(classification, indent=2, sort_keys=True), flush=True)
    return OpenProblemsResult(classification, scib, manifest, artifacts)
