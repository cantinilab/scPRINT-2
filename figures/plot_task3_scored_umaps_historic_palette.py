#!/usr/bin/env python3
"""Render the scored Task3 UMAP coordinates with the historical figure palette."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COORDINATES = (
    REPO_ROOT
    / "data/results/cross_species_embedding/"
    "task3_fresh_umaps_scoring_graph_20260818/plots/"
    "task3_fresh_scored_embeddings_umaps.h5ad"
)
DEFAULT_PCA_COORDINATES = (
    REPO_ROOT
    / "data/results/cross_species_embedding/"
    "task3_pca_umap_scoring_graph_20260821/"
    "task3_pca_umap_scoring_graph.h5ad"
)
DEFAULT_OUTPUT = REPO_ROOT / "figures/task3_cross_species_umaps_scored_20260819"

METHODS = {
    "pca": {
        "title": "PCA",
        "umap_key": "X_umap_pca",
        "predicted_annotation": None,
    },
    "transcriptformer": {
        "title": "TranscriptFormer",
        "umap_key": "X_umap_transcriptformer",
        "predicted_annotation": None,
    },
    "scprint2": {
        "title": "scPRINT-2",
        "umap_key": "X_umap_scprint2",
        "predicted_annotation": (
            "scprint2_converted_predicted_cell_type_ontology"
        ),
    },
    "scprint2_ft": {
        "title": "scPRINT-2 FT",
        "umap_key": "X_umap_scprint2_ft",
        "predicted_annotation": (
            "scprint2_ft_converted_predicted_cell_type_ontology"
        ),
    },
}

ANNOTATIONS = {
    "organism": "Species",
    "converted_predicted_cell_type_ontology": (
        "Predicted cell type ontology"
    ),
    "converted_cell_type_ontology": "Converted cell type ontology",
    "cell_type": "Cell type",
}
OVERVIEW_ANNOTATIONS = (
    "converted_cell_type_ontology",
    "organism",
    "converted_predicted_cell_type_ontology",
)

# Exact order and colors from the historical plot.  These are the first 19
# colors in scanpy.pl.palettes.default_20, associated with the legend labels
# shown in the original Task3 figure.
HISTORIC_CELL_TYPE_ORDER = (
    "T cell",
    "basal cell",
    "brush cell",
    "ciliated cell",
    "club cell",
    "endothelial cell",
    "epithelial cell",
    "fibroblast",
    "goblet cell",
    "ionocyte",
    "macrophage",
    "mesenchymal cell",
    "mesothelial cell",
    "neuroendocrine cell",
    "pericyte",
    "pneumocyte",
    "type I pneumocyte",
    "type II pneumocyte",
    "unknown",
)
HISTORIC_CELL_TYPE_COLORS = dict(
    zip(HISTORIC_CELL_TYPE_ORDER, sc.pl.palettes.default_20[:19], strict=True)
)

# The organism palette is deliberately independent of the cell-type palette.
# It keeps the historical blue/orange species semantics while using shades that
# do not collide with the exact Scanpy colors reserved for cellular labels.
ORGANISM_COLORS = {
    "cat": "#005a9c",
    "tiger": "#f28e2b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=Path, default=DEFAULT_COORDINATES)
    parser.add_argument(
        "--pca-coordinates", type=Path, default=DEFAULT_PCA_COORDINATES
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def historic_family(label: str) -> str | None:
    """Map raw and ontology labels onto the historical legend families."""

    normalized = normalize_label(label)
    if normalized in {"unknown", "others", "7", "9", "nan", "none"}:
        return "unknown"
    if normalized == "atii" or re.search(
        r"(alveolar|pneumocyte).*\b(type 2|type ii)\b", normalized
    ):
        return "type II pneumocyte"
    if normalized == "ati" or re.search(
        r"(alveolar|pneumocyte).*\b(type 1|type i)\b", normalized
    ):
        return "type I pneumocyte"
    if "basal cell" in normalized:
        return "basal cell"
    if "brush cell" in normalized or "tuft cell" in normalized:
        return "brush cell"
    if "ciliated" in normalized:
        return "ciliated cell"
    if "club cell" in normalized or "clara cell" in normalized:
        return "club cell"
    if any(
        token in normalized
        for token in (
            "endothelial",
            "endocardial",
            "capillary",
            "lymphatic vessel",
            "hepatic sinusoid",
        )
    ):
        return "endothelial cell"
    if "mesothelial" in normalized:
        return "mesothelial cell"
    if "epithelial" in normalized:
        return "epithelial cell"
    if "fibroblast" in normalized:
        return "fibroblast"
    if "goblet cell" in normalized:
        return "goblet cell"
    if "ionocyte" in normalized:
        return "ionocyte"
    if any(
        token in normalized
        for token in ("macrophage", "microglial", "kupffer cell", "monocyte")
    ):
        return "macrophage"
    if any(
        token in normalized
        for token in ("mesenchymal", "mesodermal", "chondrocyte")
    ):
        return "mesenchymal cell"
    if "neuroendocrine" in normalized or "enteroendocrine" in normalized:
        return "neuroendocrine cell"
    if "pericyte" in normalized:
        return "pericyte"
    if "pneumocyte" in normalized or "pulmonary alveolar epithelial" in normalized:
        return "pneumocyte"
    if re.search(r"(^|[^a-z])t cell([^a-z]|$)", normalized):
        return "T cell"
    return None


def ordered_labels(values: pd.Series) -> list[str]:
    labels = values.astype(object).where(values.notna(), "Unknown").astype(str)
    counts = labels.value_counts()
    return sorted(counts.index, key=lambda label: (-int(counts[label]), label))


def build_cell_palette(labels: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    unmatched: list[str] = []
    for label in labels:
        family = historic_family(label)
        if family is None:
            unmatched.append(label)
        else:
            result[label] = HISTORIC_CELL_TYPE_COLORS[family]

    # Retain semantic historical colors where possible, then use Scanpy's
    # extended categorical palette deterministically for labels absent from the
    # original 19-class legend.  Colors already reserved by the historical map
    # are removed so unmatched categories remain distinguishable.
    reserved = {color.casefold() for color in HISTORIC_CELL_TYPE_COLORS.values()}
    extended = [
        color for color in sc.pl.palettes.default_102 if color.casefold() not in reserved
    ]
    if len(unmatched) > len(extended):
        raise ValueError(
            f"Need {len(unmatched)} extended colors but only {len(extended)} remain"
        )
    for label, color in zip(sorted(unmatched), extended, strict=False):
        result[label] = color
    return result


def draw_panel(
    coordinates: np.ndarray,
    labels: pd.Series,
    colors: dict[str, str],
    *,
    method_title: str,
    annotation_title: str,
    output_stem: Path,
    random_state: int,
) -> None:
    labels = labels.astype(object).where(labels.notna(), "Unknown").astype(str)
    draw_order = np.random.default_rng(random_state).permutation(len(labels))
    categories = ordered_labels(labels)

    figure_width = 12.5 if len(categories) <= 22 else 20.0
    fig, ax = plt.subplots(figsize=(figure_width, 8.0))
    for category in categories:
        selected = draw_order[labels.iloc[draw_order].to_numpy() == category]
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
    fig.subplots_adjust(
        left=0.07,
        right=0.72 if len(categories) <= 22 else 0.43,
        bottom=0.1,
        top=0.92,
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


def draw_overview(
    data: ad.AnnData,
    palettes: dict[str, dict[str, str]],
    output_dir: Path,
    random_state: int,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 14), constrained_layout=True)
    for row, annotation in enumerate(OVERVIEW_ANNOTATIONS):
        for column, (method, config) in enumerate(METHODS.items()):
            ax = axes[row, column]
            coordinates = np.asarray(data.obsm[str(config["umap_key"])])
            draw_order = np.random.default_rng(random_state).permutation(data.n_obs)
            if row == 0:
                ax.set_title(str(config["title"]), fontsize=11)
            if column == 0:
                ax.set_ylabel(ANNOTATIONS[annotation], fontsize=11)
            source = annotation
            if annotation == "converted_predicted_cell_type_ontology":
                source = config["predicted_annotation"]
            if source is None:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                continue
            labels = (
                data.obs[str(source)]
                .astype(object)
                .where(data.obs[str(source)].notna(), "Unknown")
                .astype(str)
            )
            point_colors = [palettes[annotation][value] for value in labels.iloc[draw_order]]
            ax.scatter(
                coordinates[draw_order, 0],
                coordinates[draw_order, 1],
                c=point_colors,
                s=1.5,
                alpha=0.7,
                linewidths=0,
                rasterized=True,
            )
            if column != 0:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#bdbdbd")
                spine.set_linewidth(0.5)
    fig.suptitle(
        "Task3 scored embeddings — historical annotation colors", fontsize=15
    )
    stem = output_dir / "task3_scored_embeddings_umap_overview_3x4_historic_palette"
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"), dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.coordinates.exists():
        raise FileNotFoundError(args.coordinates)
    if not args.pca_coordinates.exists():
        raise FileNotFoundError(args.pca_coordinates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(args.output_dir.iterdir())
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty {args.output_dir}; pass --overwrite"
        )

    data = ad.read_h5ad(args.coordinates)
    pca = ad.read_h5ad(args.pca_coordinates, backed="r")
    if not data.obs_names.equals(pca.obs_names):
        raise ValueError("PCA and scored-coordinate cell order differ")
    if "X_umap_pca" not in pca.obsm:
        raise KeyError(f"X_umap_pca missing from {args.pca_coordinates}")
    data.obsm["X_umap_pca"] = np.asarray(pca.obsm["X_umap_pca"], dtype=np.float32)
    pca.file.close()
    for method, config in METHODS.items():
        key = str(config["umap_key"])
        if key not in data.obsm:
            raise KeyError(f"{key!r} missing from {args.coordinates}")

    organism_labels = sorted(data.obs["organism"].astype(str).unique())
    missing_organisms = set(organism_labels) - set(ORGANISM_COLORS)
    if missing_organisms:
        raise ValueError(f"No historical organism colors for {missing_organisms}")

    cell_labels = sorted(
        {
            value
            for column in (
                "converted_cell_type_ontology",
                "cell_type",
                "scprint2_converted_predicted_cell_type_ontology",
                "scprint2_ft_converted_predicted_cell_type_ontology",
            )
            for value in data.obs[column].astype(str).unique()
        }
    )
    shared_cell_palette = build_cell_palette(cell_labels)
    palettes = {
        "organism": dict(ORGANISM_COLORS),
        "converted_predicted_cell_type_ontology": shared_cell_palette,
        "converted_cell_type_ontology": shared_cell_palette,
        "cell_type": shared_cell_palette,
    }

    panel_count = 0
    for method, config in METHODS.items():
        coordinates = np.asarray(data.obsm[str(config["umap_key"])])
        for annotation, annotation_title in ANNOTATIONS.items():
            source = annotation
            if annotation == "converted_predicted_cell_type_ontology":
                source = config["predicted_annotation"]
            if source is None:
                continue
            draw_panel(
                coordinates,
                data.obs[str(source)],
                palettes[annotation],
                method_title=str(config["title"]),
                annotation_title=annotation_title,
                output_stem=args.output_dir / f"{method}__{annotation}",
                random_state=args.random_state,
            )
            panel_count += 1
    draw_overview(data, palettes, args.output_dir, args.random_state)

    metadata = {
        "coordinates": str(args.coordinates.resolve()),
        "coordinates_sha256": file_sha256(args.coordinates),
        "pca_coordinates": str(args.pca_coordinates.resolve()),
        "pca_coordinates_sha256": file_sha256(args.pca_coordinates),
        "n_cells": data.n_obs,
        "panels": panel_count,
        "umap_coordinates_recomputed": False,
        "overview": {
            "png": "task3_scored_embeddings_umap_overview_3x4_historic_palette.png",
            "pdf": "task3_scored_embeddings_umap_overview_3x4_historic_palette.pdf",
            "rows": list(OVERVIEW_ANNOTATIONS),
            "columns": list(METHODS),
            "prediction_unavailable_for": ["pca", "transcriptformer"],
        },
        "organism_palette": ORGANISM_COLORS,
        "cell_palette": {
            label: to_hex(color) for label, color in sorted(shared_cell_palette.items())
        },
        "historical_cell_type_order": list(HISTORIC_CELL_TYPE_ORDER),
        "historical_cell_type_palette": HISTORIC_CELL_TYPE_COLORS,
        "palette_rule": (
            "organism uses an independent fixed cat/tiger palette; cell-like "
            "annotations share semantic historical Task3 colors, with Scanpy "
            "default_102 reserved for labels absent from the historical legend"
        ),
        "random_state": args.random_state,
    }
    metadata_path = args.output_dir / "task3_umap_historic_palette.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COMPLETE").touch()


if __name__ == "__main__":
    main()
