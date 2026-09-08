#!/usr/bin/env python3
"""Plot Pandora UMAPs from the exact graphs used by the matched scIB run."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

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
    "pca": {
        "title": "Expression PCA50",
        "has_predictions": False,
    },
    "scprint2": {
        "title": "scPRINT-2 zero-shot — cell-type token",
        "has_predictions": True,
    },
    "scprint2_ft": {
        "title": "scPRINT-2 FT — cell-type token, MMD 0.03",
        "has_predictions": True,
    },
    "transcriptformer": {
        "title": "TranscriptFormer Metazoa",
        "has_predictions": False,
    },
}

ANNOTATION_TITLES = {
    "converted_cell_type_ontology": "Cell type ontology",
    "species": "Organism / species (scIB batch)",
    "converted_predicted_cell_type_ontology": "Predicted cell type ontology",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-graph", type=Path, required=True)
    parser.add_argument("--scprint2-graph", type=Path, required=True)
    parser.add_argument("--scprint2-ft-graph", type=Path, required=True)
    parser.add_argument("--transcriptformer-graph", type=Path, required=True)
    parser.add_argument("--scprint2-predictions", type=Path, required=True)
    parser.add_argument("--scprint2-ft-predictions", type=Path, required=True)
    parser.add_argument("--ontology-map-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--prediction-top-n", type=int, default=24)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def sha256_strings(values: pd.Index) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_ontology_map(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an ontology mapping object in {path}")
    return {str(key): str(value) for key, value in payload.items() if value}


def converted(values: pd.Series, names: dict[str, str]) -> pd.Series:
    raw = values.fillna("unknown").astype(str)
    return raw.map(lambda value: names.get(value, value))


def load_prediction_column(path: Path, obs_names: pd.Index) -> pd.Series:
    source = ad.read_h5ad(path, backed="r")
    key = "pred_cell_type_ontology_term_id"
    if key not in source.obs:
        source.file.close()
        raise KeyError(f"Missing obs[{key!r}] in {path}")
    if source.obs_names.equals(obs_names):
        result = source.obs[key].copy()
    elif obs_names.isin(source.obs_names).all():
        result = source.obs.loc[obs_names, key].copy()
    else:
        source.file.close()
        raise ValueError(f"Prediction cells do not align with {path}")
    source.file.close()
    result.index = obs_names
    return result.astype(str)


def load_scored_graph(
    method: str,
    path: Path,
    ontology_names: dict[str, str],
    prediction_path: Path | None,
) -> tuple[ad.AnnData, dict[str, Any]]:
    data = ad.read_h5ad(path)
    required_obs = ["cell_type_ontology_term_id", "species"]
    missing = [key for key in required_obs if key not in data.obs]
    if missing:
        raise KeyError(f"Missing obs columns in {path}: {missing}")
    if "connectivities" not in data.obsp or "neighbors" not in data.uns:
        raise KeyError(f"The scored KNN graph is missing from {path}")
    if "X_scib_embedding" not in data.obsm:
        raise KeyError(f"The exact scored representation is missing from {path}")
    if data.obsp["connectivities"].shape != (data.n_obs, data.n_obs):
        raise ValueError(f"Invalid connectivity shape in {path}")

    data.obs["cell_type_ontology_term_id"] = (
        data.obs["cell_type_ontology_term_id"].astype(str).values
    )
    data.obs["converted_cell_type_ontology"] = converted(
        data.obs["cell_type_ontology_term_id"], ontology_names
    ).values
    data.obs["species"] = data.obs["species"].astype(str).values

    prediction_categories = 0
    if prediction_path is not None:
        predictions = load_prediction_column(prediction_path, data.obs_names)
        data.obs["pred_cell_type_ontology_term_id"] = predictions.values
        data.obs["converted_predicted_cell_type_ontology"] = converted(
            predictions, ontology_names
        ).values
        prediction_categories = int(predictions.nunique())

    representation = np.asarray(data.obsm["X_scib_embedding"])
    if representation.shape[0] != data.n_obs or not np.isfinite(representation).all():
        raise ValueError(f"Invalid scored representation in {path}")

    metadata = {
        "method": method,
        "title": METHODS[method]["title"],
        "scored_graph": str(path.resolve()),
        "representation_key": "X_scib_embedding",
        "representation_shape": list(representation.shape),
        "connectivities_nnz": int(data.obsp["connectivities"].nnz),
        "neighbors": data.uns["neighbors"].get("params", {}),
        "prediction_source": str(prediction_path.resolve()) if prediction_path else None,
        "prediction_categories": prediction_categories,
        "obs_names_sha256": sha256_strings(data.obs_names),
    }
    return data, metadata


def compute_umap(
    data: ad.AnnData,
    *,
    random_state: int,
    min_dist: float,
    spread: float,
) -> None:
    # Do not call sc.pp.neighbors here: the graph is the exact graph persisted
    # by the scIB scoring run for this representation.
    sc.tl.umap(
        data,
        random_state=random_state,
        min_dist=min_dist,
        spread=spread,
        init_pos="spectral",
    )
    coordinates = np.asarray(data.obsm["X_umap"])
    if coordinates.shape != (data.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError("UMAP returned invalid coordinates")


def collapse_predictions(
    values: pd.Series, top_n: int
) -> tuple[pd.Series, dict[str, Any]]:
    values = values.fillna("Unknown").astype(str)
    counts = values.value_counts()
    kept = list(counts.head(top_n).index)
    omitted = int(max(len(counts) - len(kept), 0))
    if omitted:
        other_label = f"Other predictions ({omitted} labels)"
        display = values.where(values.isin(kept), other_label)
    else:
        other_label = None
        display = values
    return display, {
        "original_categories": int(len(counts)),
        "kept_categories": kept,
        "other_label": other_label,
        "other_cells": int((display == other_label).sum()) if other_label else 0,
    }


def categorical_palette(labels: list[str], *, prediction: bool = False) -> dict[str, Any]:
    ordered = sorted(set(labels))
    if prediction:
        colors = plt.get_cmap("turbo")(np.linspace(0.03, 0.97, max(len(ordered), 1)))
    elif len(ordered) <= 10:
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(ordered), 1)))
    else:
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(ordered), 1)))
    result = {label: tuple(color) for label, color in zip(ordered, colors)}
    for label in ordered:
        if label.lower() == "unknown" or label.startswith("Other predictions"):
            result[label] = (0.65, 0.65, 0.65, 1.0)
    return result


def scatter_panel(
    ax: Any,
    coordinates: np.ndarray,
    labels: pd.Series,
    colors: dict[str, Any],
    order: np.ndarray,
) -> None:
    values = labels.fillna("Unknown").astype(str).to_numpy()
    ax.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[colors[value] for value in values[order]],
        s=1.4,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.55)


def legend_handles(labels: pd.Series, colors: dict[str, Any]) -> list[Line2D]:
    values = labels.fillna("Unknown").astype(str)
    counts = values.value_counts()
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=colors[label],
            markeredgewidth=0,
            label=f"{label} (n={int(count):,})",
        )
        for label, count in counts.items()
    ]


def palette_handles(colors: dict[str, Any]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=color,
            markeredgewidth=0,
            label=label,
        )
        for label, color in colors.items()
    ]


def plot_comparison(
    datasets: dict[str, ad.AnnData],
    methods: list[str],
    annotation: str,
    labels_by_method: dict[str, pd.Series],
    colors: dict[str, Any],
    output_stem: Path,
    random_state: int,
) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(6.0 * len(methods), 6.2))
    axes = np.atleast_1d(axes)
    order = np.random.default_rng(random_state).permutation(datasets[methods[0]].n_obs)
    for ax, method in zip(axes, methods):
        scatter_panel(
            ax,
            datasets[method].obsm["X_umap"],
            labels_by_method[method],
            colors,
            order,
        )
        ax.set_title(str(METHODS[method]["title"]), fontsize=11)
    fig.suptitle(ANNOTATION_TITLES[annotation], fontsize=14)
    handles = palette_handles(colors)
    legend_columns = 1 if len(handles) <= 18 else 2
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=7.2,
        ncol=legend_columns,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 0.995, 0.94))
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_individual(
    data: ad.AnnData,
    method: str,
    annotation: str,
    labels: pd.Series,
    colors: dict[str, Any],
    output_stem: Path,
    random_state: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    order = np.random.default_rng(random_state).permutation(data.n_obs)
    scatter_panel(ax, data.obsm["X_umap"], labels, colors, order)
    ax.set_title(
        f"{METHODS[method]['title']} — {ANNOTATION_TITLES[annotation]}", fontsize=13
    )
    handles = legend_handles(labels, colors)
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=7.2 if len(handles) > 30 else 8.0,
        ncol=1 if len(handles) <= 28 else 2,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_overview(
    datasets: dict[str, ad.AnnData],
    display_labels: dict[str, dict[str, pd.Series]],
    palettes: dict[str, dict[str, Any]],
    output_stem: Path,
    random_state: int,
) -> None:
    methods = list(METHODS)
    annotations = list(ANNOTATION_TITLES)
    fig, axes = plt.subplots(3, 4, figsize=(19, 13.5))
    order = np.random.default_rng(random_state).permutation(datasets[methods[0]].n_obs)
    for row, annotation in enumerate(annotations):
        for column, method in enumerate(methods):
            ax = axes[row, column]
            labels = display_labels.get(annotation, {}).get(method)
            if labels is None:
                ax.axis("off")
                continue
            scatter_panel(
                ax,
                datasets[method].obsm["X_umap"],
                labels,
                palettes[annotation],
                order,
            )
            if row == 0:
                ax.set_title(str(METHODS[method]["title"]), fontsize=10.5)
            if column == 0:
                ax.set_ylabel(ANNOTATION_TITLES[annotation], fontsize=10.5)
    fig.suptitle("Pandora lung — exact scored representations", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    ontology_names = read_ontology_map(args.ontology_map_json)
    graph_paths = {
        "pca": args.pca_graph,
        "scprint2": args.scprint2_graph,
        "scprint2_ft": args.scprint2_ft_graph,
        "transcriptformer": args.transcriptformer_graph,
    }
    prediction_paths = {
        "pca": None,
        "scprint2": args.scprint2_predictions,
        "scprint2_ft": args.scprint2_ft_predictions,
        "transcriptformer": None,
    }

    datasets: dict[str, ad.AnnData] = {}
    run_metadata: dict[str, Any] = {
        "protocol": {
            "graph": "exact precomputed KNN graph persisted by the scIB 1.1.3 scoring run; neighbors not recomputed",
            "cell_type": "cell_type_ontology_term_id converted with Cell Ontology names",
            "organism": "species, identical to the scIB batch key; organism_ontology_term_id is constant after mouse-space alignment",
            "predicted_cell_type": "pred_cell_type_ontology_term_id converted with Cell Ontology names",
            "prediction_top_n": args.prediction_top_n,
            "umap": {
                "random_state": args.random_state,
                "min_dist": args.min_dist,
                "spread": args.spread,
                "init_pos": "spectral",
            },
        },
        "versions": {
            "anndata": package_version("anndata"),
            "matplotlib": package_version("matplotlib"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scanpy": package_version("scanpy"),
            "umap-learn": package_version("umap-learn"),
        },
        "ontology_map": str(args.ontology_map_json.resolve()),
        "methods": {},
    }

    reference_obs_names: pd.Index | None = None
    for method in METHODS:
        data, method_metadata = load_scored_graph(
            method,
            graph_paths[method],
            ontology_names,
            prediction_paths[method],
        )
        if reference_obs_names is None:
            reference_obs_names = data.obs_names.copy()
        elif not data.obs_names.equals(reference_obs_names):
            raise ValueError(f"Cell order differs for {method}")
        compute_umap(
            data,
            random_state=args.random_state,
            min_dist=args.min_dist,
            spread=args.spread,
        )
        datasets[method] = data
        run_metadata["methods"][method] = method_metadata

    display_labels: dict[str, dict[str, pd.Series]] = {
        "converted_cell_type_ontology": {},
        "species": {},
        "converted_predicted_cell_type_ontology": {},
    }
    for method, data in datasets.items():
        display_labels["converted_cell_type_ontology"][method] = data.obs[
            "converted_cell_type_ontology"
        ].astype(str)
        display_labels["species"][method] = data.obs["species"].astype(str)
        if bool(METHODS[method]["has_predictions"]):
            collapsed, collapse_metadata = collapse_predictions(
                data.obs["converted_predicted_cell_type_ontology"],
                args.prediction_top_n,
            )
            display_labels["converted_predicted_cell_type_ontology"][method] = collapsed
            run_metadata["methods"][method]["prediction_display"] = collapse_metadata

    palettes: dict[str, dict[str, Any]] = {}
    for annotation, labels_by_method in display_labels.items():
        labels = [
            label
            for values in labels_by_method.values()
            for label in values.fillna("Unknown").astype(str).unique()
        ]
        palettes[annotation] = categorical_palette(
            labels,
            prediction=annotation == "converted_predicted_cell_type_ontology",
        )

    method_order = list(METHODS)
    plot_comparison(
        datasets,
        method_order,
        "converted_cell_type_ontology",
        display_labels["converted_cell_type_ontology"],
        palettes["converted_cell_type_ontology"],
        args.output_dir / "pandora_task3_matched_umap_cell_type_ontology",
        args.random_state,
    )
    plot_comparison(
        datasets,
        method_order,
        "species",
        display_labels["species"],
        palettes["species"],
        args.output_dir / "pandora_task3_matched_umap_species",
        args.random_state,
    )
    predicted_methods = ["scprint2", "scprint2_ft"]
    plot_comparison(
        datasets,
        predicted_methods,
        "converted_predicted_cell_type_ontology",
        display_labels["converted_predicted_cell_type_ontology"],
        palettes["converted_predicted_cell_type_ontology"],
        args.output_dir / "pandora_task3_matched_umap_predicted_cell_type_ontology",
        args.random_state,
    )

    for annotation, labels_by_method in display_labels.items():
        for method, labels in labels_by_method.items():
            plot_individual(
                datasets[method],
                method,
                annotation,
                labels,
                palettes[annotation],
                args.output_dir / f"{method}__{annotation}",
                args.random_state,
            )
    plot_overview(
        datasets,
        display_labels,
        palettes,
        args.output_dir / "pandora_task3_matched_umap_overview",
        args.random_state,
    )

    compact_obs = datasets["pca"].obs[
        ["cell_type_ontology_term_id", "converted_cell_type_ontology", "species"]
    ].copy()
    for method in ("scprint2", "scprint2_ft"):
        compact_obs[f"{method}_pred_cell_type_ontology_term_id"] = datasets[method].obs[
            "pred_cell_type_ontology_term_id"
        ].values
        compact_obs[f"{method}_converted_predicted_cell_type_ontology"] = datasets[
            method
        ].obs["converted_predicted_cell_type_ontology"].values
    compact = ad.AnnData(
        X=sparse.csr_matrix((len(compact_obs), 0), dtype=np.float32),
        obs=compact_obs,
    )
    for method, data in datasets.items():
        compact.obsm[f"X_umap_{method}"] = np.asarray(
            data.obsm["X_umap"], dtype=np.float32
        )
    coordinate_path = args.output_dir / "pandora_task3_matched_umaps.h5ad"
    compact.write_h5ad(coordinate_path, compression="lzf")

    expected_pngs = 14
    pngs = sorted(args.output_dir.glob("*.png"))
    if len(pngs) != expected_pngs or any(path.stat().st_size == 0 for path in pngs):
        raise RuntimeError(
            f"Expected {expected_pngs} non-empty PNGs, found {len(pngs)}"
        )
    run_metadata["outputs"] = {
        "n_cells": compact.n_obs,
        "coordinate_file": coordinate_path.name,
        "png_count": len(pngs),
        "pdf_count": len(list(args.output_dir.glob("*.pdf"))),
        "overview": "pandora_task3_matched_umap_overview.png",
    }
    with (args.output_dir / "pandora_task3_matched_umaps.metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(run_metadata, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    (args.output_dir / "COMPLETE").touch()


if __name__ == "__main__":
    main()
