#!/usr/bin/env python3
"""Build reproducible Task3 UMAPs from the exact freshly scored embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy import sparse

METHODS = {
    "scprint2": {
        "title": "scPRINT-2",
        "embedding_key": "scprint_emb_cell_type_ontology_term_id",
        "has_predictions": True,
    },
    "scprint2_ft": {
        "title": "scPRINT-2 FT (MMD 0.03)",
        "embedding_key": "scprint_emb_cell_type_ontology_term_id",
        "has_predictions": True,
    },
    "transcriptformer": {
        "title": "TranscriptFormer Metazoa",
        "embedding_key": "model_emb",
        "has_predictions": False,
    },
}

ANNOTATIONS = {
    "organism": "Organism (Task3 batch: orig.ident)",
    "converted_predicted_cell_type_ontology": (
        "Converted predicted cell type ontology"
    ),
    "converted_cell_type_ontology": "Converted cell type ontology",
    "cell_type": "Cell type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scprint2", type=Path, required=True)
    parser.add_argument("--scprint2-ft", type=Path, required=True)
    parser.add_argument("--transcriptformer", type=Path, required=True)
    parser.add_argument("--ontology-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--neighbors-random-state", type=int, default=0)
    parser.add_argument("--umap-random-state", type=int, default=42)
    return parser.parse_args()


def sha256_strings(values: pd.Index) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def ontology_name_map(path: Path) -> dict[str, str]:
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            result = {str(key): str(value) for key, value in json.load(handle).items()}
    else:
        ontology = pd.read_parquet(path)
        if ontology.index.name != "ontology_id":
            if "ontology_id" not in ontology.columns:
                raise ValueError(f"No ontology_id field in {path}")
            ontology = ontology.set_index("ontology_id")
        if "name" not in ontology.columns:
            raise ValueError(f"No name field in {path}")
        result = ontology["name"].dropna().astype(str).to_dict()
    result["unknown"] = "Unknown"
    return result


def converted(values: pd.Series, names: dict[str, str]) -> pd.Series:
    raw = values.astype(object).where(values.notna(), "unknown").astype(str)
    return raw.map(lambda value: names.get(value, value)).astype("category")


def string_labels(values: pd.Series, missing: str = "Unknown") -> pd.Series:
    return values.astype(object).where(values.notna(), missing).astype(str)


def load_method(
    method: str,
    path: Path,
    ontology_names: dict[str, str],
) -> tuple[ad.AnnData, dict[str, object]]:
    config = METHODS[method]
    source = ad.read_h5ad(path, backed="r")
    embedding_key = str(config["embedding_key"])
    if embedding_key not in source.obsm:
        raise KeyError(f"{embedding_key!r} missing from {path}")

    required_obs = ["orig.ident", "cell_type_ontology_term_id", "cell_type"]
    if bool(config["has_predictions"]):
        required_obs.append("pred_cell_type_ontology_term_id")
    missing = [column for column in required_obs if column not in source.obs]
    if missing:
        raise KeyError(f"Missing obs columns in {path}: {missing}")

    obs = pd.DataFrame(index=source.obs_names.copy())
    obs["organism"] = source.obs["orig.ident"].astype(str).values
    obs["cell_type_ontology_term_id"] = (
        source.obs["cell_type_ontology_term_id"].astype(str).values
    )
    obs["converted_cell_type_ontology"] = (
        converted(source.obs["cell_type_ontology_term_id"], ontology_names)
        .astype(str)
        .values
    )
    obs["cell_type"] = source.obs["cell_type"].astype(str).values
    if bool(config["has_predictions"]):
        obs["pred_cell_type_ontology_term_id"] = (
            source.obs["pred_cell_type_ontology_term_id"].astype(str).values
        )
        obs["converted_predicted_cell_type_ontology"] = (
            converted(source.obs["pred_cell_type_ontology_term_id"], ontology_names)
            .astype(str)
            .values
        )

    embedding = np.asarray(source.obsm[embedding_key], dtype=np.float32)
    source.file.close()
    if embedding.shape[0] != len(obs):
        raise ValueError(
            f"Embedding row mismatch for {method}: {embedding.shape[0]} != {len(obs)}"
        )

    plot_data = ad.AnnData(
        X=sparse.csr_matrix((len(obs), 0), dtype=np.float32),
        obs=obs,
    )
    plot_data.obsm["X_embedding"] = embedding
    metadata = {
        "method": method,
        "title": config["title"],
        "input": str(path.resolve()),
        "embedding_key": embedding_key,
        "embedding_shape": list(embedding.shape),
        "obs_names_sha256": sha256_strings(plot_data.obs_names),
    }
    return plot_data, metadata


def compute_umap(
    data: ad.AnnData,
    *,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    neighbors_random_state: int,
    umap_random_state: int,
) -> None:
    # This mirrors the paper evaluator's direct neighbor construction on the
    # integrated embedding: sc.pp.neighbors(..., use_rep=<embedding key>).
    sc.pp.neighbors(
        data,
        n_neighbors=n_neighbors,
        use_rep="X_embedding",
        metric=metric,
        random_state=neighbors_random_state,
    )
    sc.tl.umap(data, min_dist=min_dist, random_state=umap_random_state)


def palette(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    ordered = sorted(labels, key=lambda value: (value.lower() == "unknown", value))
    if len(ordered) <= 10:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(ordered), 1)))
    elif len(ordered) <= 20:
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(ordered)))
    else:
        colors = plt.get_cmap("gist_ncar")(np.linspace(0.02, 0.98, len(ordered)))
    result = {label: tuple(color) for label, color in zip(ordered, colors)}
    if "Unknown" in result:
        result["Unknown"] = (0.55, 0.55, 0.55, 1.0)
    if "unknown" in result:
        result["unknown"] = (0.55, 0.55, 0.55, 1.0)
    return result


def plot_one(
    coordinates: np.ndarray,
    labels: pd.Series,
    colors: dict[str, tuple[float, float, float, float]],
    *,
    method_title: str,
    annotation_title: str,
    output_stem: Path,
    random_state: int,
) -> None:
    labels = string_labels(labels)
    order = np.random.default_rng(random_state).permutation(len(labels))
    categories = sorted(
        labels.unique(), key=lambda value: (-int((labels == value).sum()), value)
    )

    fig, ax = plt.subplots(figsize=(12.5, 8.0), constrained_layout=True)
    for category in categories:
        selected = order[labels.iloc[order].to_numpy() == category]
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=4,
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            color=colors[category],
        )
    ax.set_title(f"{method_title} — {annotation_title}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.6)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=colors[category],
            markeredgewidth=0,
            label=f"{category} (n={(labels == category).sum():,})",
        )
        for category in categories
    ]
    legend_columns = (
        1 if len(categories) <= 22 else min(4, int(np.ceil(len(categories) / 24)))
    )
    ax.legend(
        handles=handles,
        title=f"{len(categories)} categories",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
        fontsize=6.5 if len(categories) > 40 else 8,
        title_fontsize=8,
        ncol=legend_columns,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_contact_sheet(
    datasets: dict[str, ad.AnnData],
    palettes: dict[str, dict[str, tuple[float, float, float, float]]],
    output_dir: Path,
    random_state: int,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
    annotation_order = list(ANNOTATIONS)
    for row, (method, data) in enumerate(datasets.items()):
        coordinates = data.obsm["X_umap"]
        order = np.random.default_rng(random_state).permutation(data.n_obs)
        for column, annotation in enumerate(annotation_order):
            ax = axes[row, column]
            if annotation not in data.obs:
                ax.axis("off")
                continue
            labels = string_labels(data.obs[annotation])
            colors = palettes[annotation]
            point_colors = [colors[value] for value in labels.iloc[order]]
            ax.scatter(
                coordinates[order, 0],
                coordinates[order, 1],
                c=point_colors,
                s=1.5,
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
            if row == 0:
                ax.set_title(ANNOTATIONS[annotation], fontsize=11)
            if column == 0:
                ax.set_ylabel(str(METHODS[method]["title"]), fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#bdbdbd")
                spine.set_linewidth(0.5)
    fig.suptitle("Task3 fresh scored embeddings — UMAP overview", fontsize=15)
    fig.savefig(output_dir / "task3_fresh_scored_embeddings_umap_overview.png", dpi=220)
    fig.savefig(output_dir / "task3_fresh_scored_embeddings_umap_overview.pdf", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {args.output_dir}")
    ontology_names = ontology_name_map(args.ontology_parquet)
    paths = {
        "scprint2": args.scprint2,
        "scprint2_ft": args.scprint2_ft,
        "transcriptformer": args.transcriptformer,
    }

    datasets: dict[str, ad.AnnData] = {}
    metadata: dict[str, object] = {
        "protocol": {
            "neighbors": {
                "n_neighbors": args.n_neighbors,
                "metric": args.metric,
                "use_rep": "exact scored embedding, no PCA",
                "random_state": args.neighbors_random_state,
            },
            "umap": {
                "min_dist": args.min_dist,
                "random_state": args.umap_random_state,
            },
            "organism_column": "orig.ident",
            "ontology_source": str(args.ontology_parquet.resolve()),
        },
        "methods": {},
    }
    reference_obs_names: pd.Index | None = None
    for method, path in paths.items():
        data, method_metadata = load_method(method, path, ontology_names)
        if reference_obs_names is None:
            reference_obs_names = data.obs_names.copy()
        elif not data.obs_names.equals(reference_obs_names):
            raise ValueError(f"Cell order differs for {method}")
        compute_umap(
            data,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            neighbors_random_state=args.neighbors_random_state,
            umap_random_state=args.umap_random_state,
        )
        datasets[method] = data
        metadata["methods"][method] = method_metadata

    palettes: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    for annotation in ANNOTATIONS:
        labels = sorted(
            {
                value
                for data in datasets.values()
                if annotation in data.obs
                for value in string_labels(data.obs[annotation]).unique()
            }
        )
        palettes[annotation] = palette(labels)

    for method, data in datasets.items():
        for annotation, annotation_title in ANNOTATIONS.items():
            if annotation not in data.obs:
                continue
            plot_one(
                data.obsm["X_umap"],
                data.obs[annotation],
                palettes[annotation],
                method_title=str(METHODS[method]["title"]),
                annotation_title=annotation_title,
                output_stem=args.output_dir / f"{method}__{annotation}",
                random_state=args.umap_random_state,
            )
    plot_contact_sheet(datasets, palettes, args.output_dir, args.umap_random_state)

    compact_obs = (
        datasets["scprint2"]
        .obs[
            [
                "organism",
                "cell_type_ontology_term_id",
                "converted_cell_type_ontology",
                "cell_type",
            ]
        ]
        .copy()
    )
    compact_obs["scprint2_converted_predicted_cell_type_ontology"] = (
        datasets["scprint2"].obs["converted_predicted_cell_type_ontology"].values
    )
    compact_obs["scprint2_ft_converted_predicted_cell_type_ontology"] = (
        datasets["scprint2_ft"].obs["converted_predicted_cell_type_ontology"].values
    )
    compact = ad.AnnData(
        X=sparse.csr_matrix((len(compact_obs), 0), dtype=np.float32),
        obs=compact_obs,
    )
    for method, data in datasets.items():
        compact.obsm[f"X_umap_{method}"] = data.obsm["X_umap"].astype(np.float32)
    compact.write_h5ad(args.output_dir / "task3_fresh_scored_embeddings_umaps.h5ad")

    metadata["outputs"] = {
        "n_cells": compact.n_obs,
        "panels": 11,
        "coordinate_file": "task3_fresh_scored_embeddings_umaps.h5ad",
        "overview": "task3_fresh_scored_embeddings_umap_overview.png",
    }
    with (args.output_dir / "task3_fresh_scored_embeddings_umaps.metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (args.output_dir / "COMPLETE").touch()


if __name__ == "__main__":
    main()
