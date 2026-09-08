from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.collect_eye4_scib113_scores import (
    TOTAL_FORMULA,
    collect_scores,
    parse_input_spec,
)

METHODS = ["PCA", "Random", "scPRINT-ZS", "scPRINT-FT", "TranscriptFormer"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    root: Path, method: str, *, metadata_change: tuple[str, object] | None = None
) -> Path:
    method_root = root / method
    method_root.mkdir(parents=True)
    graph = method_root / "graphs" / f"{method}.h5ad"
    graph.parent.mkdir()
    graph.write_bytes(f"graph-{method}".encode())
    score = method_root / "score.csv"
    pd.DataFrame(
        [
            {
                "Method": method,
                "Batch correction": 0.25,
                "Bio conservation": 0.75,
                "Total": 0.55,
                "ARI": 0.4,
            }
        ]
    ).to_csv(score, index=False)
    metadata = {
        "protocol": "scib 1.1.3 with precomputed graphs",
        "versions": {"scib": "1.1.3", "scanpy": "1.9.1"},
        "source": "/remote/source.h5ad",
        "source_sha256": "source-hash",
        "batch_key": "species",
        "label_key": "celltype",
        "cells": 100,
        "labels": 13,
        "batches": 4,
        "expression_preprocessing": ["pca", "neighbors"],
        "expression_connectivities_sha256": "expression-graph-hash",
        "total_formula": TOTAL_FORMULA,
        "embeddings": {
            method: {
                "embedding": f"/remote/{method}.h5ad",
                "embedding_sha256": f"embedding-{method}",
                "embedding_key": "X_emb",
                "precomputed_graph": str(graph),
                "precomputed_graph_sha256": _sha256(graph),
                "connectivities_sha256_before_scib": f"knn-{method}",
                "connectivities_sha256_after_scib": f"knn-{method}",
            }
        },
    }
    if metadata_change:
        metadata[metadata_change[0]] = metadata_change[1]
    score.with_suffix(".metadata.json").write_text(json.dumps(metadata) + "\n")
    score.with_suffix(".COMPLETE").write_text("COMPLETE\n")
    return score


def _five_inputs(tmp_path: Path) -> list[tuple[str, Path]]:
    return [(method, _write_artifact(tmp_path / "inputs", method)) for method in METHODS]


def test_collect_scores_combines_five_methods_and_copies_graphs(tmp_path) -> None:
    inputs = _five_inputs(tmp_path)
    output = tmp_path / "collected" / "scores.csv"

    table, metadata = collect_scores(inputs[::-1], METHODS, output)

    assert table["Method"].tolist() == METHODS
    assert pd.read_csv(output)["Method"].tolist() == METHODS
    assert metadata["method_order"] == METHODS
    assert list(metadata["embeddings"]) == METHODS
    for method, provenance in metadata["embeddings"].items():
        graph = Path(provenance["precomputed_graph"])
        assert graph == (output.parent / "precomputed_graphs" / f"{method}.h5ad")
        assert graph.read_bytes() == f"graph-{method}".encode()
        assert provenance["precomputed_graph_sha256"] == _sha256(graph)
    stored = json.loads(output.with_suffix(".metadata.json").read_text())
    assert stored["source_sha256"] == "source-hash"
    assert len(stored["validated_individual_artifacts"]) == 5
    assert output.with_suffix(".COMPLETE").read_text() == "COMPLETE\n"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("versions", {"scib": "1.1.2"}),
        ("source_sha256", "different"),
        ("batch_key", "donor"),
        ("label_key", "other_label"),
        ("cells", 99),
        ("labels", 12),
        ("batches", 3),
        ("expression_connectivities_sha256", "different-graph"),
    ],
)
def test_collect_scores_rejects_inconsistent_metadata(
    tmp_path, key: str, value: object
) -> None:
    inputs = _five_inputs(tmp_path)
    changed = _write_artifact(
        tmp_path / "changed", METHODS[-1], metadata_change=(key, value)
    )
    inputs[-1] = (METHODS[-1], changed)

    with pytest.raises(RuntimeError, match=f"metadata for {key}"):
        collect_scores(inputs, METHODS, tmp_path / "scores.csv")


def test_collect_scores_rejects_wrong_total(tmp_path) -> None:
    inputs = _five_inputs(tmp_path)
    score = inputs[0][1]
    table = pd.read_csv(score)
    table.loc[0, "Total"] = 0.1
    table.to_csv(score, index=False)

    with pytest.raises(RuntimeError, match="Total score formula"):
        collect_scores(inputs, METHODS, tmp_path / "scores.csv")


def test_collect_scores_validates_methods_and_refuses_overwrite(tmp_path) -> None:
    inputs = _five_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="missing=.*TranscriptFormer"):
        collect_scores(inputs[:-1], METHODS, tmp_path / "missing.csv")

    output = tmp_path / "scores.csv"
    output.write_text("existing\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collect_scores(inputs, METHODS, output)


def test_parse_input_spec() -> None:
    assert parse_input_spec("PCA=/tmp/pca.csv") == ("PCA", Path("/tmp/pca.csv"))
    with pytest.raises(Exception, match="METHOD=CSV"):
        parse_input_spec("missing-separator")
