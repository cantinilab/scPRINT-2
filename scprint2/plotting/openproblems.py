"""OpenProblems score loading, preparation, and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASETS = ["dkd", "gtex_v9", "hypomap", "mouse_pancreas_atlas"]
CLASSIFICATION_DATASET_LABELS = {
    "dkd": "dkd (cell types: 13)",
    "gtex_v9": "gtex_v9 (cell types: 53)",
    "hypomap": "hypomap (cell types: 13)",
    "mouse_pancreas_atlas": "mouse_pancreas_atlas (cell types: 11)",
}
BATCH_DATASET_LABELS = {
    "dkd": "dkd (11 donors, 2 assays)",
    "gtex_v9": "gtex_v9 (16 donors, 1 assay)",
    "hypomap": "hypomap (24 donors, 4 assays)",
    "mouse_pancreas_atlas": "mouse_pancreas_atlas (56 donors, 2 assays)",
}
CLASSIFICATION_METHOD_ORDER = [
    "majority_vote",
    "random_labels",
    "knn",
    "logistic_regression",
    "mlp",
    "xgboost",
    "naive_bayes",
    "scanvi",
    "scanvi_scarches",
    "seurat_transferdata",
    "singler",
    "scgpt_zeroshot",
    "scimilarity_knn",
    "scimilarity",
    "scPRINT-1 (zero-shot)",
    "scPRINT-2 (zero-shot)",
    "scPRINT-2 (knn)",
    "scPRINT-2 (fine-tune)",
]
CLASSIFICATION_HIDDEN_METHODS = {"uce"}
BATCH_METHOD_ORDER = [
    "no_integration_batch",
    "no_integration",
    "shuffle_integration",
    "shuffle_integration_by_batch",
    "shuffle_integration_by_cell_type",
    "embed_cell_types",
    "embed_cell_types_jittered",
    "scanorama",
    "batchelor_fastmnn",
    "combat",
    "harmonypy",
    "scalex",
    "scanvi",
    "pyliger",
    "liger",
    "scvi",
    "uce",
    "scgpt_zeroshot",
    "scimilarity",
    "TranscriptFormer",
    "scPRINT-1",
    "scPRINT-2 ZS cell_type",
    "scPRINT-2-FT",
]
BATCH_HIDDEN_METHODS = {"geneformer", "scPRINT-2"}
BIO_METRICS = ["ari", "nmi", "isolated_label_asw", "clisi", "asw_label"]
BATCH_METRICS = ["pcr", "graph_connectivity", "asw_batch", "ilisi", "kbet"]

CLASSIFICATION_DISPLAY_NAMES = {
    "scgpt_zeroshot": "scGPT (kNN)",
    "scimilarity_knn": "SCimilarity (kNN)",
    "scimilarity": "SCimilarity (kNN 9M atlas)",
    "scPRINT-1 (zero-shot)": "scPRINT-1 (zero-shot)",
    "scPRINT-2 (zero-shot)": "scPRINT-2 (zero-shot)",
}
BATCH_DISPLAY_NAMES = {
    "scgpt_zeroshot": "scGPT",
    "scPRINT-1": "scPRINT-1 (zero-shot)",
    "scPRINT-2 ZS cell_type": "scPRINT-2 (zero-shot)",
    "scPRINT-2-FT": "scPRINT-2 (fine-tune)",
}


def _validate_datasets(frame: pd.DataFrame, source: Path) -> None:
    observed = set(frame["dataset"] if "dataset" in frame else frame["method"])
    missing = sorted(set(DATASETS) - observed)
    if missing:
        raise ValueError(f"{source} is missing datasets: {missing}")


def load_classification_tables(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path)
    raw = raw.rename(columns={"method": "dataset"})
    _validate_datasets(raw, path)
    method_columns = [
        column for column in raw.columns if column not in {"score", "dataset"}
    ]
    known_methods = set(CLASSIFICATION_METHOD_ORDER) | CLASSIFICATION_HIDDEN_METHODS
    unknown = sorted(set(method_columns) - known_methods)
    if unknown:
        raise ValueError(
            f"Add new classification methods to the display order: {unknown}"
        )

    score_rows = {
        "f1_macro": "macro_f1",
        "f1_weighted": "weighted",
        "accuracy": "accuracy",
        # For single-label multiclass classification, micro-F1 equals accuracy.
        "f1_micro": "accuracy",
    }
    tables = {}
    display_order = [
        method for method in CLASSIFICATION_METHOD_ORDER if method in method_columns
    ]
    for output_metric, sheet_metric in score_rows.items():
        selected = raw.loc[raw["score"] == sheet_metric].set_index("dataset")
        selected = selected.reindex(DATASETS)[method_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        tables[output_metric] = selected.T.reindex(display_order)
    return tables


def _fixed_denominator_mean(
    metrics: pd.DataFrame, metric_names: list[str]
) -> pd.DataFrame:
    selected = metrics.reindex(metric_names)
    means = selected.sum(axis=0, skipna=True, min_count=1) / len(metric_names)
    return means


def load_batch_tables(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path)
    _validate_datasets(raw, path)
    method_columns = [
        column for column in raw.columns if column not in {"dataset", "metric"}
    ]
    known_methods = set(BATCH_METHOD_ORDER) | BATCH_HIDDEN_METHODS
    unknown = sorted(set(method_columns) - known_methods)
    if unknown:
        raise ValueError(f"Add new batch methods to the display order: {unknown}")
    raw[method_columns] = raw[method_columns].apply(pd.to_numeric, errors="coerce")

    bio_by_dataset = {}
    batch_by_dataset = {}
    for dataset in DATASETS:
        metrics = raw.loc[raw["dataset"] == dataset].set_index("metric")[method_columns]
        bio_by_dataset[dataset] = _fixed_denominator_mean(metrics, BIO_METRICS)
        batch_by_dataset[dataset] = _fixed_denominator_mean(metrics, BATCH_METRICS)

    bio = pd.DataFrame(bio_by_dataset).reindex(BATCH_METHOD_ORDER)
    batch = pd.DataFrame(batch_by_dataset).reindex(BATCH_METHOD_ORDER)
    total = 0.6 * bio + 0.4 * batch
    return {"Total": total, "Bio conservation": bio, "Batch correction": batch}


def prepare_scores(
    raw_scores: pd.DataFrame, *, use_zscore: bool, exclude_missing: bool
) -> pd.DataFrame:
    scores = raw_scores.apply(pd.to_numeric, errors="coerce").copy()
    if not exclude_missing:
        scores = scores.fillna(0)
    if use_zscore:
        means = scores.mean(axis=0, skipna=True)
        standard_deviations = scores.std(axis=0, skipna=True, ddof=0)
        scores = scores.subtract(means, axis="columns")
        variable = standard_deviations[standard_deviations > 0].index
        scores.loc[:, variable] = scores.loc[:, variable].divide(
            standard_deviations.loc[variable], axis="columns"
        )
        constant = standard_deviations[standard_deviations == 0].index
        scores.loc[:, constant] = scores.loc[:, constant].where(
            scores.loc[:, constant].isna(), 0.0
        )
    return scores


def plot_score_distribution(
    raw_scores: pd.DataFrame,
    *,
    score_name: str,
    title: str,
    filename: str,
    output_dir: Path,
    dataset_labels: dict[str, str],
    use_zscore: bool,
    exclude_missing: bool,
    separators: set[str],
    figure_size: tuple[int, int],
    display_names: dict[str, str],
) -> None:
    raw_numeric = raw_scores.apply(pd.to_numeric, errors="coerce")
    scores = prepare_scores(
        raw_scores, use_zscore=use_zscore, exclude_missing=exclude_missing
    )
    method_order = [
        method for method in scores.index if scores.loc[method].notna().any()
    ]
    scores = scores.reindex(method_order)
    positions = np.arange(len(method_order), dtype=float)
    dataset_order = list(scores.columns)
    offsets = np.linspace(-0.12, 0.12, len(dataset_order))
    colors = sns.color_palette("deep", len(dataset_order))

    fig, ax = plt.subplots(figsize=figure_size)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for dataset_index, dataset in enumerate(dataset_order):
        values = scores[dataset].to_numpy(dtype=float)
        raw_values = raw_numeric.reindex(method_order)[dataset].to_numpy(dtype=float)
        # Zero and missing scores still affect penalized calculations, but
        # neither is drawn as a dataset dot.
        available = ~np.isnan(raw_values) & ~np.isclose(raw_values, 0.0)
        ax.scatter(
            positions[available] + offsets[dataset_index],
            values[available],
            s=95,
            color=colors[dataset_index],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            label=dataset_labels[dataset],
            zorder=3,
        )

    if not use_zscore:
        if exclude_missing:
            aggregate_label = "Average (available datasets)"
        else:
            aggregate_label = "Penalized average"
        ax.scatter(
            positions,
            scores.mean(axis="columns", skipna=True),
            s=105,
            marker="D",
            color="black",
            edgecolor="white",
            linewidth=0.8,
            label=aggregate_label,
            zorder=5,
        )

    for method_index, method in enumerate(method_order):
        if method_index and method in separators:
            ax.axvline(
                method_index - 0.5,
                color="#222222",
                linewidth=1.2,
                linestyle=(0, (1, 3)),
                zorder=1,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [display_names.get(method, method) for method in method_order],
        rotation=42,
        ha="right",
    )
    for label in ax.get_xticklabels():
        if label.get_text().startswith("scPRINT"):
            label.set_fontweight("bold")
            label.set_color("#1565c0")

    score_scale = "z-score" if use_zscore else "raw score"
    policy = "missing scores excluded" if exclude_missing else "missing scores set to 0"
    fig.suptitle(
        f"{title}\ndots = {score_scale} per dataset; {policy}",
        fontweight="bold",
        y=0.99,
    )
    ax.set_xlabel("Methods", fontweight="bold", labelpad=16)
    ax.set_ylabel(score_name, fontweight="bold")
    ax.grid(False)
    ax.tick_params(axis="x", length=0)
    sns.despine(ax=ax)
    if not use_zscore:
        ax.set_ylim(0.0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        borderaxespad=0.2,
        ncol=3,
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=1.0,
        fontsize="small",
        handletextpad=0.45,
        columnspacing=0.9,
    )
    # Keep the title and legend in a dedicated figure band above the axes so
    # neither can cover observations near the top of the score range.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_openproblems_plots(input_dir: Path, output_root: Path) -> int:
    sns.set_theme(style="whitegrid", context="talk")
    classification_tables = load_classification_tables(
        input_dir / "fig3_openproblems_classification_scores.csv"
    )
    batch_tables = load_batch_tables(input_dir / "fig5_op_scib_scores.csv")

    # Both requested views penalize missing scores as zero. The raw view shows
    # the penalized mean; the z-score view intentionally has no mean marker.
    variants = [
        (False, False, "raw_nan_as_zero"),
        (True, False, "zscore_nan_as_zero"),
    ]
    classification_specs = {
        "f1_weighted": "Weighted F1",
        "f1_micro": "Micro F1",
        "f1_macro": "Macro F1",
        "accuracy": "Accuracy",
    }
    batch_specs = {
        "Total": ("scIB total score", "batch_total"),
        "Bio conservation": ("scIB bio-conservation score", "batch_bio_conservation"),
        "Batch correction": ("scIB batch-correction score", "batch_correction"),
    }

    for use_zscore, exclude_missing, variant in variants:
        output_dir = output_root / variant
        for metric, display_name in classification_specs.items():
            plot_score_distribution(
                classification_tables[metric],
                score_name=display_name,
                title=f"OpenProblems label projection: {display_name}",
                filename=f"classification_{metric}",
                output_dir=output_dir,
                dataset_labels=CLASSIFICATION_DATASET_LABELS,
                use_zscore=use_zscore,
                exclude_missing=exclude_missing,
                separators={"knn", "scgpt_zeroshot"},
                figure_size=(14, 10),
                display_names=CLASSIFICATION_DISPLAY_NAMES,
            )
        for score_name, (axis_label, filename) in batch_specs.items():
            plot_score_distribution(
                batch_tables[score_name],
                score_name=axis_label,
                title=f"OpenProblems batch integration: {score_name}",
                filename=filename,
                output_dir=output_dir,
                dataset_labels=BATCH_DATASET_LABELS,
                use_zscore=use_zscore,
                exclude_missing=exclude_missing,
                separators={
                    "scanorama",
                    "uce",
                },
                figure_size=(16, 11),
                display_names=BATCH_DISPLAY_NAMES,
            )

    return len(variants) * (len(classification_specs) + len(batch_specs))
