import json

import pandas as pd

from scprint2.evaluation.result_collection import (
    collect_pandora_scib113_scores,
    collect_pandora_task3_matched_scores,
)


def test_collect_pandora_scib113_scores_compares_legacy_and_modern(tmp_path):
    methods = {"demo": ("legacy-method", "modern-method")}
    score_root = tmp_path / "pandora_lung_filtered_93423_scib113_demo"
    pd.DataFrame(
        [
            {
                "Method": "legacy-method",
                "Total": 0.7,
                "Batch correction": 0.6,
                "Bio conservation": 0.8,
            }
        ]
    ).to_csv(score_root.with_suffix(".csv"), index=False)
    score_root.with_suffix(".metadata.json").write_text(
        json.dumps({"source": "demo"}) + "\n"
    )
    score_root.with_suffix(".COMPLETE").write_text("COMPLETE\n")
    modern_path = tmp_path / "modern.tsv"
    pd.DataFrame(
        [
            {
                "Embedding": "modern-method",
                "Total": 0.5,
                "Batch correction": 0.4,
                "Bio conservation": 0.6,
            }
        ]
    ).to_csv(modern_path, sep="\t", index=False)

    legacy, comparison, metadata = collect_pandora_scib113_scores(
        tmp_path, modern_path, methods=methods
    )

    assert legacy["Method"].tolist() == ["legacy-method"]
    assert metadata == {"legacy-method": {"source": "demo"}}
    row = comparison.iloc[0]
    assert row["Method"] == "legacy-method"
    assert row["legacy_minus_modern"] == 0.19999999999999996
    assert row["scib_1.1.3_batch"] == 0.6
    assert row["scib_metrics_bio"] == 0.6


def _task3_metadata(method: str) -> dict:
    return {
        "protocol": "task3-matched",
        "versions": {"scib": "1.1.3"},
        "source": "source.h5ad",
        "source_sha256": "sha",
        "batch_key": "species",
        "label_key": "cell_type_ontology_term_id",
        "cells": 10,
        "labels": 2,
        "batches": 2,
        "expression_preprocessing": "cp10k-log1p",
        "expression_connectivities_sha256": "graph-sha",
        "total_formula": "0.4 * Batch correction + 0.6 * Bio conservation",
        "embeddings": {
            method: {
                "connectivities_sha256_before_scib": "graph-sha",
                "connectivities_sha256_after_scib": "graph-sha",
            }
        },
    }


def test_collect_pandora_task3_matched_scores_orders_and_validates(tmp_path):
    method_order = ["method-b", "method-a"]
    for method, batch, bio in [
        ("method-a", 0.25, 0.5),
        ("method-b", 0.5, 0.75),
    ]:
        root = tmp_path / method
        total = 0.4 * batch + 0.6 * bio
        pd.DataFrame(
            [
                {
                    "Method": method,
                    "Total": total,
                    "Batch correction": batch,
                    "Bio conservation": bio,
                }
            ]
        ).to_csv(root.with_suffix(".csv"), index=False)
        root.with_suffix(".metadata.json").write_text(
            json.dumps(_task3_metadata(method)) + "\n"
        )
        root.with_suffix(".COMPLETE").write_text("COMPLETE\n")

    table, metadata = collect_pandora_task3_matched_scores(
        tmp_path, method_order=method_order
    )

    assert table["Method"].tolist() == method_order
    assert metadata["versions"] == {"scib": "1.1.3"}
    assert metadata["batch_key"] == "species"
    assert metadata["method_order"] == method_order
    assert len(metadata["validated_individual_metadata"]) == 2
