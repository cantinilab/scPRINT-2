"""Persist reusable per-cell outputs from OpenProblems classification runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from anndata import AnnData


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_classification_output(
    embedded: AnnData,
    output_path: str | Path,
    *,
    metadata: dict[str, Any],
) -> None:
    """Save compact per-cell predictions, logits, and embeddings to H5AD."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prediction_columns = [
        column
        for column in embedded.obs.columns
        if column.startswith("pred_")
    ]
    required_columns = [
        "cell_type",
        "cell_type_ontology_term_id",
        "donor_id",
        "leiden",
        "seurat_clusters",
        "classification_held_out",
    ]
    obs_columns = list(
        dict.fromkeys(
            column
            for column in [*required_columns, *prediction_columns]
            if column in embedded.obs
        )
    )
    output = AnnData(obs=embedded.obs.loc[:, obs_columns].copy())

    embedding_keys = [
        key
        for key in embedded.obsm
        if key.startswith("scprint_emb") or key.startswith("classification_")
    ]
    for key in embedding_keys:
        output.obsm[key] = np.asarray(embedded.obsm[key], dtype=np.float32)

    output.uns["artifact_schema"] = "scprint_op_classification_output_v1"
    output.uns["embedding_keys"] = embedding_keys
    if "classification_logit_labels" in embedded.uns:
        output.uns["classification_logit_labels"] = list(
            embedded.uns["classification_logit_labels"]
        )
    output.write_h5ad(output_path, compression="gzip")

    sidecar = output_path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "artifact": str(output_path),
                "embedding_keys": embedding_keys,
                "classification_logit_labels": _jsonable(
                    embedded.uns.get("classification_logit_labels", [])
                ),
                "n_cells": int(output.n_obs),
                "obs_columns": obs_columns,
                **_jsonable(metadata),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
