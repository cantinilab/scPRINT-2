"""Reusable label-projection scoring helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

CELL_TYPE = "cell_type_ontology_term_id"


def _decode(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode()
    return str(value)


def read_h5ad_obs_column(obs: h5py.Group, key: str) -> np.ndarray:
    """Read a dense or categorical H5AD obs column without loading the AnnData."""
    node = obs[key]
    if isinstance(node, h5py.Group):
        codes = node["codes"][:]
        categories = np.asarray(
            [_decode(value) for value in node["categories"][:]], dtype=object
        )
        values = np.empty(codes.shape, dtype=object)
        valid = codes >= 0
        values[~valid] = ""
        values[valid] = categories[codes[valid]]
        return values
    values = node[:]
    if values.dtype.kind in {"S", "O", "U"}:
        return np.asarray([_decode(value) for value in values], dtype=object)
    return values


def load_saved_arrays(
    input_path: str | Path,
    *,
    truth_key: str,
    held_out_key: str,
    prediction_key: str,
    embedding_keys: list[str],
) -> dict[str, np.ndarray]:
    """Load only the columns and embeddings required by the evaluation."""
    with h5py.File(input_path, "r") as handle:
        obs = handle["obs"]
        arrays = {
            "truth": read_h5ad_obs_column(obs, truth_key).astype(str),
            "held_out": read_h5ad_obs_column(obs, held_out_key).astype(bool),
            "checkpoint_prediction": read_h5ad_obs_column(obs, prediction_key).astype(
                str
            ),
        }
        for key in embedding_keys:
            if key not in handle["obsm"]:
                raise KeyError(f"Embedding key {key!r} is absent from {input_path}")
            arrays[key] = handle["obsm"][key][:].astype(np.float32, copy=False)
    return arrays


def load_descendants(
    metadata_path: str | Path, class_key: str = CELL_TYPE
) -> dict[str, set[str]]:
    """Convert the saved integer decoder hierarchy into ontology-label sets."""
    metadata = json.loads(Path(metadata_path).read_text())
    decoder = {
        int(index): label
        for index, label in metadata["label_decoders"][class_key].items()
    }
    hierarchy = metadata["labels_hierarchy"].get(class_key, {})
    return {
        decoder[int(parent)]: {
            decoder[int(child)] for child in children if int(child) in decoder
        }
        for parent, children in hierarchy.items()
        if int(parent) in decoder
    }


def _contingency(
    predicted: np.ndarray, truth: np.ndarray
) -> tuple[list[str], list[str], np.ndarray]:
    predicted_levels = sorted(set(predicted))
    truth_levels = sorted(set(truth))
    table = pd.crosstab(
        pd.Series(truth, name="truth"),
        pd.Series(predicted, name="predicted"),
    ).reindex(index=truth_levels, columns=predicted_levels, fill_value=0)
    return truth_levels, predicted_levels, table.to_numpy(dtype=np.int64)


def _jaccard_similarity(counts: np.ndarray) -> np.ndarray:
    truth_counts = counts.sum(axis=1, keepdims=True)
    predicted_counts = counts.sum(axis=0, keepdims=True)
    union = truth_counts + predicted_counts - counts
    return np.divide(
        counts,
        union,
        out=np.zeros_like(counts, dtype=np.float64),
        where=union != 0,
    )


def _semantic_mapping(
    predicted_levels: list[str],
    truth_levels: list[str],
    descendants: dict[str, set[str]] | None,
) -> dict[str, str]:
    """Pin exact labels and, optionally, the most specific true ancestor."""
    truth_set = set(truth_levels)
    mapping = {label: label for label in predicted_levels if label in truth_set}
    if descendants is None:
        return mapping
    for predicted in predicted_levels:
        if predicted in mapping:
            continue
        candidates = [
            truth
            for truth in truth_levels
            if predicted in descendants.get(truth, set())
        ]
        if candidates:
            mapping[predicted] = min(
                candidates,
                key=lambda truth: (len(descendants.get(truth, set())), truth),
            )
    return mapping


def fit_direct_jaccard_mapping(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    descendants: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    """Fit the current OpenProblems scPRINT direct Jaccard mapping."""
    truth_levels, predicted_levels, counts = _contingency(predicted, truth)
    similarity = _jaccard_similarity(counts)
    mapping = _semantic_mapping(predicted_levels, truth_levels, descendants)
    for column, predicted_label in enumerate(predicted_levels):
        if predicted_label not in mapping:
            mapping[predicted_label] = truth_levels[int(similarity[:, column].argmax())]
    return mapping


def fit_iterative_hungarian_mapping(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    descendants: dict[str, set[str]] | None = None,
) -> dict[str, str]:
    """Fit the iterative Jaccard/Hungarian mapping used by SCimilarity."""
    truth_levels, predicted_levels, counts = _contingency(predicted, truth)
    distance = 1.0 - _jaccard_similarity(counts)
    mapping = _semantic_mapping(predicted_levels, truth_levels, descendants)
    predicted_index = {label: index for index, label in enumerate(predicted_levels)}
    while len(mapping) < len(predicted_levels):
        remaining = [label for label in predicted_levels if label not in mapping]
        columns = [predicted_index[label] for label in remaining]
        truth_indices, local_predicted_indices = linear_sum_assignment(
            distance[:, columns]
        )
        for truth_index, local_predicted_index in zip(
            truth_indices, local_predicted_indices, strict=True
        ):
            mapping[remaining[int(local_predicted_index)]] = truth_levels[
                int(truth_index)
            ]
    return mapping


def apply_mapping(predicted: np.ndarray, mapping: dict[str, str]) -> np.ndarray:
    """Apply a training-derived mapping; unseen test predictions remain unmapped."""
    return np.asarray([mapping.get(label, label) for label in predicted], dtype=object)


def hierarchy_correct(
    predicted: np.ndarray,
    truth: np.ndarray,
    descendants: dict[str, set[str]],
) -> np.ndarray:
    """Return whether a prediction is exact or a descendant of the true label."""
    return np.asarray(
        [
            pred == true or pred in descendants.get(true, set())
            for pred, true in zip(predicted, truth, strict=True)
        ],
        dtype=bool,
    )


def score_predictions(
    predicted: np.ndarray,
    truth: np.ndarray,
    descendants: dict[str, set[str]],
) -> dict[str, float]:
    """Compute exact OpenProblems metrics plus hierarchy-aware accuracy."""
    return {
        "accuracy_exact": float(np.mean(predicted == truth)),
        "accuracy_hierarchy": float(
            np.mean(hierarchy_correct(predicted, truth, descendants))
        ),
        "f1_macro_exact": float(f1_score(truth, predicted, average="macro")),
        "f1_weighted_exact": float(f1_score(truth, predicted, average="weighted")),
    }


def nearest_neighbor_indices(
    train_embedding: np.ndarray,
    test_embedding: np.ndarray,
    *,
    n_neighbors: int,
    device: str,
    block_size: int,
) -> np.ndarray:
    """Find Euclidean neighbors on GPU by blocks, with a sklearn CPU fallback."""
    if device == "cuda":
        train_tensor = torch.from_numpy(train_embedding).to("cuda")
        blocks = []
        with torch.no_grad():
            for start in range(0, len(test_embedding), block_size):
                test_tensor = torch.from_numpy(
                    test_embedding[start : start + block_size]
                ).to("cuda")
                distances = torch.cdist(test_tensor, train_tensor)
                blocks.append(
                    torch.topk(
                        distances,
                        k=n_neighbors,
                        dim=1,
                        largest=False,
                        sorted=True,
                    )
                    .indices.cpu()
                    .numpy()
                )
        return np.concatenate(blocks, axis=0)
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="minkowski",
        p=2,
        n_jobs=-1,
    ).fit(train_embedding)
    return neighbors.kneighbors(test_embedding, return_distance=False)


def vote_neighbor_labels(
    neighbor_indices: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_neighbors: int,
) -> np.ndarray:
    """Apply deterministic uniform majority voting to precomputed neighbors."""
    classes, encoded = np.unique(train_labels, return_inverse=True)
    neighbor_classes = encoded[neighbor_indices[:, :n_neighbors]]
    predictions = np.empty(len(neighbor_classes), dtype=object)
    for row, values in enumerate(neighbor_classes):
        predictions[row] = classes[np.bincount(values, minlength=len(classes)).argmax()]
    return predictions.astype(str)


def evaluate(
    arrays: dict[str, np.ndarray],
    descendants: dict[str, set[str]],
    embedding_keys: list[str],
    *,
    knn_device: str,
    knn_block_size: int,
    run_mappings: bool = True,
    run_knn: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run mappings and reference-neighbor projections on one saved output."""
    held_out = arrays["held_out"]
    train = ~held_out
    truth_train = arrays["truth"][train]
    truth_test = arrays["truth"][held_out]
    checkpoint_train = arrays["checkpoint_prediction"][train]
    checkpoint_test = arrays["checkpoint_prediction"][held_out]
    predictions = pd.DataFrame(
        {
            "input_row": np.flatnonzero(held_out),
            "truth": truth_test,
            "checkpoint_prediction": checkpoint_test,
        }
    )

    results: dict[str, Any] = {
        "n_cells": int(len(held_out)),
        "n_train": int(train.sum()),
        "n_test": int(held_out.sum()),
        "n_train_labels": int(len(set(truth_train))),
        "n_test_labels": int(len(set(truth_test))),
        "unseen_test_labels": sorted(set(truth_test) - set(truth_train)),
        "label_mappings": {},
        "methods": {
            "checkpoint_direct": score_predictions(
                checkpoint_test, truth_test, descendants
            )
        },
    }

    mapping_specs = {
        "jaccard": (fit_direct_jaccard_mapping, None),
        "jaccard_hierarchy_first": (fit_direct_jaccard_mapping, descendants),
        "hungarian_scimilarity": (fit_iterative_hungarian_mapping, None),
        "hungarian_scimilarity_hierarchy_first": (
            fit_iterative_hungarian_mapping,
            descendants,
        ),
    }
    for name, (fit_mapping, semantic_descendants) in (
        mapping_specs.items() if run_mappings else ()
    ):
        mapping = fit_mapping(
            checkpoint_train,
            truth_train,
            descendants=semantic_descendants,
        )
        predicted = apply_mapping(checkpoint_test, mapping)
        scores = score_predictions(predicted, truth_test, descendants)
        scores["n_mapped_checkpoint_labels"] = int(len(mapping))
        scores["n_unmapped_test_cells"] = int(
            sum(label not in mapping for label in checkpoint_test)
        )
        results["label_mappings"][name] = mapping
        results["methods"][name] = scores
        predictions[name] = predicted

    for embedding_key in embedding_keys if run_knn else ():
        embedding = arrays[embedding_key]
        safe_key = embedding_key.replace("scprint_emb_", "").replace(
            "scprint_emb", "full"
        )
        neighbor_indices = nearest_neighbor_indices(
            embedding[train],
            embedding[held_out],
            n_neighbors=50,
            device=knn_device,
            block_size=knn_block_size,
        )
        for method_name, neighbors in (
            ("scprint_knn_k50", 50),
            ("scprint_knn_k10", 10),
        ):
            predicted = vote_neighbor_labels(
                neighbor_indices,
                truth_train,
                n_neighbors=neighbors,
            )
            output_name = f"{method_name}_{safe_key}"
            results["methods"][output_name] = score_predictions(
                predicted, truth_test, descendants
            )
            predictions[output_name] = predicted
    return results, predictions
