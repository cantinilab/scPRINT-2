"""OpenProblems benchmark helpers for corrected scPRINT-1 embeddings."""

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
from scprint2.tasks import Embedder
from scprint2.tasks.cell_emb import compute_classification

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
        "test_donors": ["mouse_pancreatic_islet_atlas_Hrovatin__VSG__MUC13639"],
    },
}


@dataclass(frozen=True)
class OpenProblemsV1Config:
    """Python-facing configuration for corrected scPRINT-1 evaluation."""

    dataset: str
    input: Path | str
    checkpoint: Path | str
    output_dir: Path | str
    num_workers: int = 8
    seed: int = 0
    classification_only: bool = False

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"Unknown OpenProblems dataset: {self.dataset}")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("input", "checkpoint", "output_dir"):
            values[key] = str(values[key])
        return values

    def as_namespace(self) -> argparse.Namespace:
        return argparse.Namespace(**asdict(self))


@dataclass
class OpenProblemsV1Result:
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
    adata.obs[f"{prediction_column}_direct"] = adata.obs[prediction_column].copy()
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


def load_scprint1_model(checkpoint: str):
    model = scPRINT2.load_from_checkpoint(
        checkpoint,
        precpt_gene_emb=None,
        gene_pos_file=None,
    )
    if not torch.cuda.is_available():
        model = model.to(torch.float32)
    return model.to("cuda" if torch.cuda.is_available() else "cpu")


def load_openproblems_v1_dataset(
    config: OpenProblemsV1Config | argparse.Namespace,
):
    """Load a scPRINT-1 benchmark input and validate its held-out donor."""
    args = config.as_namespace() if isinstance(config, OpenProblemsV1Config) else config
    adata = sc.read_h5ad(args.input)
    missing = sorted({"donor_id", CELL_TYPE} - set(adata.obs))
    if missing:
        raise KeyError(f"Missing OpenProblems obs columns: {missing}")
    test_donors = DATASETS[args.dataset]["test_donors"]
    if not adata.obs["donor_id"].isin(test_donors).any():
        raise ValueError(f"No cells found for held-out donors {test_donors}")
    return adata


def summarize_openproblems_v1_dataset(
    adata, config: OpenProblemsV1Config | argparse.Namespace
) -> pd.Series:
    """Return the scPRINT-1 split summary shown by its notebook."""
    args = config.as_namespace() if isinstance(config, OpenProblemsV1Config) else config
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
        },
        name="OpenProblems scPRINT-1 dataset audit",
    )


def embed_scprint1(model, adata, num_workers: int):
    """Compute and PCA-reduce the corrected scPRINT-1 cell-type token."""
    embedded, _ = Embedder(
        how="random expr",
        max_len=2_300,
        num_workers=num_workers,
        pred_embedding=[CELL_TYPE],
        doplot=False,
    )(model, adata)
    embedded.obsm["scprint_emb_cell_type_raw"] = np.asarray(
        embedded.obsm["scprint_emb"], dtype=np.float32
    ).copy()
    embedded.obsm["scprint_emb"] = pca50(embedded.obsm["scprint_emb"])
    return embedded


def build_manifest(
    args: argparse.Namespace,
    *,
    spec: dict[str, Any],
    embedded,
) -> dict[str, Any]:
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "classification_only": args.classification_only,
        "dataset": spec["name"],
        "embedding": "cell_type PCA50",
        "embedding_width": int(embedded.obsm["scprint_emb"].shape[1]),
        "seed": args.seed,
        "test_donors": spec["test_donors"],
    }


def run_openproblems_v1_benchmark(
    config: OpenProblemsV1Config | argparse.Namespace,
) -> OpenProblemsV1Result:
    args = config.as_namespace() if isinstance(config, OpenProblemsV1Config) else config
    set_seed(args.seed)
    spec = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = load_openproblems_v1_dataset(args)
    model = load_scprint1_model(args.checkpoint)
    embedded = embed_scprint1(model, adata, args.num_workers)
    classification = classification_scores(embedded, model, spec["test_donors"])

    stem = f"{args.dataset}_scprint1_corrected_cell_type_pca50"
    manifest = build_manifest(args, spec=spec, embedded=embedded)
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
    manifest_path = output_dir / f"{stem}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    artifacts = {
        "classification_output": classification_output,
        "classification_scores": classification_path,
        "manifest": manifest_path,
    }
    if args.classification_only:
        print(json.dumps(classification, indent=2, sort_keys=True), flush=True)
        return OpenProblemsV1Result(classification, None, manifest, artifacts)

    prepare_op_scib_environment()
    result = compute_op_scib_metrics(
        embedded,
        embedding_key="scprint_emb",
        batch_key="donor_id",
        label_key="cell_type",
        solution=load_op_solution(spec["name"]),
        method_id="scPRINT-1 corrected cell_type PCA50",
    )
    scib_path = save_op_scib_result(result, output_dir / f"{stem}_op_scib.csv")
    artifacts["scib_scores"] = scib_path
    print(result.to_string(), flush=True)
    print(json.dumps(classification, indent=2, sort_keys=True), flush=True)
    return OpenProblemsV1Result(classification, result, manifest, artifacts)
