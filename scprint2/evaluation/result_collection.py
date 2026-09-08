"""Reusable result collection helpers for Task3 and Pandora score outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PANDORA_SCIB113_METHODS = {
    "ft-cell": (
        "scPRINT-2-FT-cell-token-mmd003-no-organism-classification-knn",
        "FT MMD 0.03 no organism classification cell type",
    ),
    "ft-all": (
        "scPRINT-2-FT-all-tokens-mmd003-no-organism-classification-knn",
        "FT MMD 0.03 no organism classification all",
    ),
    "zs-cell": ("scPRINT-2-ZS-cell-token-knn", "ZS cell type"),
    "zs-all": ("scPRINT-2-ZS-all-tokens-pca50", "ZS all"),
    "transcriptformer": ("TranscriptFormer", "TranscriptFormer"),
}

PANDORA_TASK3_MATCHED_METHOD_ORDER = [
    "scPRINT-2-FT-cell-token-mmd003-task3-matched",
    "scPRINT-2-FT-all-tokens-except-assay-organism-pca50-task3-matched",
    "scPRINT-2-ZS-cell-token-task3-matched",
    "scPRINT-2-ZS-all-tokens-except-assay-organism-pca50-task3-matched",
    "TranscriptFormer-Metazoa-task3-matched",
    "PCA-expression-CP10K-log1p-task3-matched",
    "Random-seed42-task3-matched",
]

BATCH_METRICS = [
    "Batch.ARI",
    "Batch.ASW",
    "Graph.iLISI",
    "Batch.NMI",
    "kBET",
    "PCR.batch",
    "Graph.connectivity",
]
BIO_METRICS = [
    "Cell.type.ARI",
    "Cell.type.ASW",
    "Graph.cLISI",
    "Cell.type.NMI",
    "HVG.conservation",
    "Trajectory.conservation",
]

# These metrics are populated for every method in the published task3 table and
# for embedding-based methods evaluated by the deposited scripts. Restricting
# the unscaled aggregate to this intersection avoids silently giving methods
# different denominators because some paper methods do not report ASW, PCR, or
# HVG conservation.
COMMON_BATCH_METRICS = [
    "Batch.ARI",
    "Graph.iLISI",
    "Batch.NMI",
    "kBET",
    "Graph.connectivity",
]
COMMON_BIO_METRICS = [
    "Cell.type.ARI",
    "Graph.cLISI",
    "Cell.type.NMI",
    "Trajectory.conservation",
]


def collect_pandora_scib113_scores(
    data_root: Path,
    modern_scores_path: Path,
    *,
    methods: dict[str, tuple[str, str]] = PANDORA_SCIB113_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    legacy_rows = []
    metadata = {}
    for slug, (legacy_name, _) in methods.items():
        root = data_root / ("pandora_lung_filtered_93423_scib113_{}".format(slug))
        score_path = root.with_suffix(".csv")
        metadata_path = root.with_suffix(".metadata.json")
        complete_path = root.with_suffix(".COMPLETE")
        for path in (score_path, metadata_path, complete_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError("Missing completed artifact: {}".format(path))
        score = pd.read_csv(score_path)
        if len(score) != 1 or score.loc[0, "Method"] != legacy_name:
            raise RuntimeError("Unexpected score row in {}".format(score_path))
        legacy_rows.append(score.iloc[0])
        metadata[legacy_name] = json.loads(metadata_path.read_text())

    legacy = pd.DataFrame(legacy_rows).sort_values("Total", ascending=False)
    modern = pd.read_csv(modern_scores_path, sep="\t")
    comparisons = []
    for _, (legacy_name, modern_name) in methods.items():
        old = legacy.loc[legacy["Method"] == legacy_name]
        new = modern.loc[modern["Embedding"] == modern_name]
        if len(old) != 1 or len(new) != 1:
            raise RuntimeError(
                "Cannot uniquely compare {!r} with {!r}".format(
                    legacy_name, modern_name
                )
            )
        comparisons.append(
            {
                "Method": legacy_name,
                "scib_1.1.3_total": float(old.iloc[0]["Total"]),
                "scib_metrics_total": float(new.iloc[0]["Total"]),
                "legacy_minus_modern": float(old.iloc[0]["Total"])
                - float(new.iloc[0]["Total"]),
                "scib_1.1.3_batch": float(old.iloc[0]["Batch correction"]),
                "scib_metrics_batch": float(new.iloc[0]["Batch correction"]),
                "scib_1.1.3_bio": float(old.iloc[0]["Bio conservation"]),
                "scib_metrics_bio": float(new.iloc[0]["Bio conservation"]),
            }
        )
    comparison = pd.DataFrame(comparisons).sort_values(
        "scib_1.1.3_total", ascending=False
    )
    return legacy, comparison, metadata


def collect_pandora_task3_matched_scores(
    score_dir: Path,
    *,
    method_order: list[str] = PANDORA_TASK3_MATCHED_METHOD_ORDER,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    metadata = []
    score_paths = sorted(score_dir.glob("*.csv"))
    for path in score_paths:
        table = pd.read_csv(path)
        if len(table) != 1:
            raise RuntimeError(f"Expected one method in {path}, found {len(table)}")
        meta_path = path.with_suffix(".metadata.json")
        complete_path = path.with_suffix(".COMPLETE")
        if not meta_path.exists() or not complete_path.exists():
            raise RuntimeError(f"Incomplete score artifact: {path}")
        rows.append(table.iloc[0].to_dict())
        metadata.append(json.loads(meta_path.read_text()))

    table = pd.DataFrame(rows)
    if set(table.get("Method", [])) != set(method_order):
        missing = sorted(set(method_order) - set(table.get("Method", [])))
        extra = sorted(set(table.get("Method", [])) - set(method_order))
        raise RuntimeError(f"Method mismatch; missing={missing}, extra={extra}")
    table["_order"] = table["Method"].map({v: i for i, v in enumerate(method_order)})
    table = table.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    reference = metadata[0]
    invariant_keys = [
        "versions",
        "source",
        "source_sha256",
        "batch_key",
        "label_key",
        "cells",
        "labels",
        "batches",
        "expression_preprocessing",
        "expression_connectivities_sha256",
        "total_formula",
    ]
    for item in metadata[1:]:
        for key in invariant_keys:
            if item[key] != reference[key]:
                raise RuntimeError(f"Inconsistent score metadata for {key}")
    if reference["versions"]["scib"] != "1.1.3":
        raise RuntimeError("Pandora task3-matched collection requires scib==1.1.3")
    if reference["batch_key"] != "species":
        raise RuntimeError("Pandora batch key must be species")
    if reference["label_key"] != "cell_type_ontology_term_id":
        raise RuntimeError("Pandora label key must be cell_type_ontology_term_id")
    expected = 0.4 * table["Batch correction"] + 0.6 * table["Bio conservation"]
    if not np.allclose(table["Total"], expected, rtol=0, atol=1e-12):
        raise RuntimeError("Total score formula validation failed")
    for item in metadata:
        for method, provenance in item["embeddings"].items():
            if (
                provenance["connectivities_sha256_before_scib"]
                != provenance["connectivities_sha256_after_scib"]
            ):
                raise RuntimeError(f"scIB changed the KNN graph for {method}")

    combined_meta = {
        "protocol": reference["protocol"],
        "versions": reference["versions"],
        "source": reference["source"],
        "source_sha256": reference["source_sha256"],
        "batch_key": reference["batch_key"],
        "label_key": reference["label_key"],
        "cells": reference["cells"],
        "labels": reference["labels"],
        "batches": reference["batches"],
        "expression_preprocessing": reference["expression_preprocessing"],
        "expression_connectivities_sha256": reference[
            "expression_connectivities_sha256"
        ],
        "total_formula": reference["total_formula"],
        "method_order": method_order,
        "validated_individual_metadata": [
            str(path.with_suffix(".metadata.json").resolve()) for path in score_paths
        ],
    }
    return table, combined_meta


def collect_task3_paper_raw(root: Path, method: str) -> dict[str, object]:
    evaluation = root / "output" / "evaluation" / "saturn"
    ari = pd.read_table(evaluation / "task3_saturn_ARI.txt").iloc[-1]
    asw = pd.read_csv(evaluation / "task3_saturn_ASW_metric.csv").iloc[-1]
    nmi = pd.read_csv(evaluation / "task3_saturn_NMI.csv").iloc[0]
    ilisi = pd.read_table(evaluation / "task3_saturn_lisi_batch_40.txt", index_col=0)
    clisi = pd.read_table(evaluation / "task3_saturn_lisi_celltype_40.txt", index_col=0)
    scib = pd.read_csv(evaluation / "task3_saturn_scib_output.csv", index_col=0).iloc[
        :, 0
    ]
    return {
        "Task": "task3",
        "Method": method,
        "Batch.ARI": 1 - max(float(ari["ari_batch"]), 0),
        "Batch.ASW": float(asw["asw_batch_norm_sub"]),
        "Graph.iLISI": float((ilisi.iloc[:-1, 0] - 1).mean()),
        "Batch.NMI": 1 - float(nmi["nmi_value_batch"]),
        "kBET": float(scib["kBET"]),
        "PCR.batch": float(scib["PCR_batch"]),
        "Graph.connectivity": float(scib["graph_conn"]),
        "Cell.type.ARI": float(ari["ari_celltype"]),
        "Cell.type.ASW": float(asw["asw_celltype_norm"]),
        "Graph.cLISI": float((12 - clisi.iloc[:-1, 0]).mean() / 11),
        "Cell.type.NMI": float(nmi["nmi_value_celltype"]),
        "HVG.conservation": float(scib["hvg_overlap"]),
        "Trajectory.conservation": float(scib["trajectory"]),
    }


def aggregate_task3_paper_scores(
    raw: pd.DataFrame, anchors: pd.DataFrame | None = None
) -> pd.DataFrame:
    anchors = raw if anchors is None else anchors
    scaled = raw[BATCH_METRICS + BIO_METRICS].copy()
    for column in scaled:
        values = scaled[column]
        anchor_values = anchors[column]
        minimum = anchor_values.min(skipna=True)
        span = anchor_values.max(skipna=True) - minimum
        scaled[column] = (values - minimum) / span if span else values
    result = raw[["Task", "Method"]].copy()
    result["Batch.Correction"] = scaled[BATCH_METRICS].mean(axis=1, skipna=True)
    result["Bio.conservation"] = scaled[BIO_METRICS].mean(axis=1, skipna=True)
    result["Overall.Score"] = (
        0.4 * result["Batch.Correction"] + 0.6 * result["Bio.conservation"]
    )
    return result.join(raw[BATCH_METRICS + BIO_METRICS]).sort_values(
        "Overall.Score", ascending=False
    )


def aggregate_task3_raw_common_scores(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the common raw metrics without any cross-method scaling."""
    required = COMMON_BATCH_METRICS + COMMON_BIO_METRICS
    missing_columns = set(required) - set(raw.columns)
    if missing_columns:
        raise KeyError(f"Missing common raw metrics: {sorted(missing_columns)}")
    if raw[required].isna().any().any():
        missing = raw.loc[raw[required].isna().any(axis=1), ["Method", *required]]
        raise ValueError(f"Common raw metrics contain missing values:\n{missing}")
    result = raw[["Task", "Method"]].copy()
    result["Raw.Batch.Mean"] = raw[COMMON_BATCH_METRICS].mean(axis=1)
    result["Raw.Bio.Mean"] = raw[COMMON_BIO_METRICS].mean(axis=1)
    result["Raw.Overall"] = (
        0.4 * result["Raw.Batch.Mean"] + 0.6 * result["Raw.Bio.Mean"]
    )
    return result.join(raw[BATCH_METRICS + BIO_METRICS]).sort_values(
        "Raw.Overall", ascending=False
    )


def load_task3_reference_raw(reference_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.read_csv(reference_path)
    raw = reference.drop(
        columns=["Overall.Score", "Batch.Correction", "Bio.conservation"]
    )
    return reference, raw


def build_task3_suite_raw(
    reference_path: Path,
    result_specs: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference, raw = load_task3_reference_raw(reference_path)
    additions = []
    for specification in result_specs:
        method, separator, root = specification.partition("=")
        if not separator or not method or not root:
            raise ValueError(
                f"Invalid --result {specification!r}; expected METHOD=ROOT"
            )
        additions.append(collect_task3_paper_raw(Path(root), method))
    raw = pd.concat([raw, pd.DataFrame(additions)], ignore_index=True)
    return reference, raw
