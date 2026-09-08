#!/usr/bin/env python3
"""Plot eye benchmark UMAPs from the exact graphs scored by scIB 1.1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

METHODS = {
    "PCA": "Expression PCA50",
    "Random": "Random seed 42\nnegative control",
    "scPRINT-2 ZS": "scPRINT-2 zero-shot\ncell-type token",
    "scPRINT-2 FT": "scPRINT-2 FT\ncell-type token, MMD 0.03",
    "TranscriptFormer": "TranscriptFormer Metazoa",
    "scPRINT-2 ZS all-except-organism": (
        "scPRINT-2 zero-shot\nall tokens except organism, PCA50"
    ),
}
REQUIRED_METHODS = {
    "PCA",
    "Random",
    "scPRINT-2 ZS",
    "scPRINT-2 FT",
    "TranscriptFormer",
}


def parse_graph(value: str) -> tuple[str, Path]:
    method, separator, path = value.partition("=")
    if not separator or method not in METHODS or not path:
        raise argparse.ArgumentTypeError(
            "graph must be one of 'PCA=PATH', 'Random=PATH', "
            "'scPRINT-2 ZS=PATH', 'scPRINT-2 FT=PATH', or "
            "'TranscriptFormer=PATH', or "
            "'scPRINT-2 ZS all-except-organism=PATH'"
        )
    return method, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="append", type=parse_graph, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=1.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unavailable"


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _load_graph(path: Path) -> ad.AnnData:
    data = ad.read_h5ad(path)
    missing_obs = {"celltype", "species"} - set(data.obs.columns)
    if missing_obs:
        raise KeyError(f"{path} lacks obs columns: {sorted(missing_obs)}")
    for key in ("connectivities", "distances"):
        if key not in data.obsp:
            raise KeyError(f"{path} lacks obsp[{key!r}]")
    if "neighbors" not in data.uns or "X_scib_embedding" not in data.obsm:
        raise KeyError(f"{path} is not a persisted scIB scoring graph")
    representation = np.asarray(data.obsm["X_scib_embedding"])
    if representation.shape[0] != data.n_obs or not np.isfinite(representation).all():
        raise ValueError(f"Invalid scored representation in {path}")
    return data


def _palette(values: pd.Series) -> dict[str, tuple[float, ...]]:
    labels = sorted(values.astype(str).unique())
    cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    return {
        label: tuple(color)
        for label, color in zip(labels, cmap(np.linspace(0, 1, len(labels))))
    }


def _scatter(
    ax,
    coordinates: np.ndarray,
    labels: pd.Series,
    palette: dict[str, tuple[float, ...]],
    order: np.ndarray,
) -> None:
    values = labels.astype(str).to_numpy()
    ax.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[palette[value] for value in values[order]],
        s=1.8,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bdbdbd")
        spine.set_linewidth(0.55)


def _legend(values: pd.Series, palette: dict[str, tuple[float, ...]]):
    counts = values.astype(str).value_counts()
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=palette[label],
            markeredgewidth=0,
            label=f"{label} (n={int(count):,})",
        )
        for label, count in counts.items()
    ]


def _comparison(
    datasets: dict[str, ad.AnnData],
    annotation: str,
    palette: dict[str, tuple[float, ...]],
    output: Path,
    random_state: int,
) -> None:
    methods = list(datasets)
    fig, axes = plt.subplots(1, len(methods), figsize=(5.1 * len(methods), 5.8))
    order = np.random.default_rng(random_state).permutation(
        datasets[methods[0]].n_obs
    )
    for ax, method in zip(axes, methods):
        _scatter(
            ax,
            np.asarray(datasets[method].obsm["X_umap"]),
            datasets[method].obs[annotation],
            palette,
            order,
        )
        ax.set_title(METHODS[method], fontsize=10.5)
    title = "Expert cell type" if annotation == "celltype" else "Species (scIB batch)"
    fig.suptitle(f"Four-species eye integration — {title}", fontsize=14)
    handles = _legend(datasets[methods[0]].obs[annotation], palette)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.995, 0.5),
        frameon=False,
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0, 0, 0.995, 0.93))
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def _overview(
    datasets: dict[str, ad.AnnData],
    palettes: dict[str, dict[str, tuple[float, ...]]],
    output: Path,
    random_state: int,
) -> None:
    methods = list(datasets)
    annotations = ("celltype", "species")
    fig, axes = plt.subplots(2, len(methods), figsize=(4.8 * len(methods), 9.2))
    order = np.random.default_rng(random_state).permutation(
        datasets[methods[0]].n_obs
    )
    for row, annotation in enumerate(annotations):
        for column, method in enumerate(methods):
            ax = axes[row, column]
            _scatter(
                ax,
                np.asarray(datasets[method].obsm["X_umap"]),
                datasets[method].obs[annotation],
                palettes[annotation],
                order,
            )
            if row == 0:
                ax.set_title(METHODS[method], fontsize=10.5)
            if column == 0:
                ax.set_ylabel(
                    "Expert cell type" if annotation == "celltype" else "Species",
                    fontsize=10.5,
                )
    fig.suptitle(
        "Four-species eye integration — exact scIB 1.1.3 scoring graphs",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    graph_paths = dict(args.graph)
    missing = REQUIRED_METHODS - set(graph_paths)
    extra = set(graph_paths) - set(METHODS)
    if missing or extra:
        raise ValueError(f"Invalid graph set; missing={sorted(missing)}, extra={sorted(extra)}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    datasets = {method: _load_graph(path) for method, path in graph_paths.items()}
    reference_names = datasets["PCA"].obs_names
    for method, data in datasets.items():
        if not data.obs_names.equals(reference_names):
            raise ValueError(f"Cell order differs for {method}")
        sc.tl.umap(
            data,
            random_state=args.random_state,
            min_dist=args.min_dist,
            spread=args.spread,
            init_pos="spectral",
        )
        coordinates = np.asarray(data.obsm["X_umap"])
        if coordinates.shape != (data.n_obs, 2) or not np.isfinite(coordinates).all():
            raise ValueError(f"Invalid UMAP coordinates for {method}")

    palettes = {
        annotation: _palette(datasets["PCA"].obs[annotation])
        for annotation in ("celltype", "species")
    }
    _comparison(
        datasets,
        "celltype",
        palettes["celltype"],
        args.output_dir / "eye4_scored_umaps_celltype",
        args.random_state,
    )
    _comparison(
        datasets,
        "species",
        palettes["species"],
        args.output_dir / "eye4_scored_umaps_species",
        args.random_state,
    )
    _overview(
        datasets,
        palettes,
        args.output_dir / "eye4_scored_umaps_overview",
        args.random_state,
    )
    metadata = {
        "protocol": "UMAP from exact precomputed KNN graphs persisted by the scib 1.1.3 run; neighbors not recomputed",
        "umap": {
            "random_state": args.random_state,
            "min_dist": args.min_dist,
            "spread": args.spread,
            "init_pos": "spectral",
        },
        "methods": {
            method: {
                "graph": str(path.resolve()),
                "sha256": _sha256(path),
                "neighbors": datasets[method].uns["neighbors"].get("params", {}),
            }
            for method, path in graph_paths.items()
        },
        "versions": {
            name: _package_version(name)
            for name in ("anndata", "matplotlib", "numpy", "pandas", "scanpy", "umap-learn")
        },
    }
    (args.output_dir / "eye4_scored_umaps.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    (args.output_dir / "UMAPS.COMPLETE").write_text("COMPLETE\n")


if __name__ == "__main__":
    main()
