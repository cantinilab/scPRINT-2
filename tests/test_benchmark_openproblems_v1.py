from argparse import Namespace
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scprint2.benchmark import openproblems_v1 as op_v1
from scripts import scprint1_op_corrected as cli


def _args(**overrides):
    values = {
        "checkpoint": "checkpoint.ckpt",
        "classification_only": False,
        "seed": 11,
    }
    values.update(overrides)
    return Namespace(**values)


def test_classification_scores_preserves_direct_predictions(monkeypatch):
    prediction_column = f"pred_{op_v1.CELL_TYPE}"
    adata = ad.AnnData(
        X=np.zeros((3, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["train", "test", "test"],
                prediction_column: ["CL:1", "CL:2", "CL:2"],
            },
            index=["cell-a", "cell-b", "cell-c"],
        ),
    )
    model = SimpleNamespace(label_decoders={}, labels_hierarchy={})

    def fake_compute(view, keys, *, label_decoders, labels_hierarchy):
        assert keys == [op_v1.CELL_TYPE]
        return {op_v1.CELL_TYPE: {"n_obs": view.n_obs}}

    monkeypatch.setattr(op_v1, "compute_classification", fake_compute)

    scores = op_v1.classification_scores(adata, model, ["test"])

    assert scores == {
        "all_cells_direct": {"n_obs": 3},
        "held_out_direct": {"n_obs": 2},
        "held_out_cells": 2,
    }
    assert adata.obs["classification_held_out"].tolist() == [False, True, True]
    assert adata.obs[f"{prediction_column}_direct"].tolist() == [
        "CL:1",
        "CL:2",
        "CL:2",
    ]


def test_classification_scores_rejects_missing_held_out_donors(monkeypatch):
    adata = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["train-a", "train-b"],
                f"pred_{op_v1.CELL_TYPE}": ["CL:1", "CL:2"],
            }
        ),
    )

    with pytest.raises(ValueError, match="No cells found"):
        op_v1.classification_scores(
            adata,
            SimpleNamespace(label_decoders={}, labels_hierarchy={}),
            ["test"],
        )


def test_pca50_and_manifest_capture_embedding_metadata():
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    pca = op_v1.pca50(values)
    assert pca.dtype == np.float32
    assert pca.shape == (4, 3)

    embedded = ad.AnnData(X=np.zeros((4, 1)))
    embedded.obsm["scprint_emb"] = pca
    spec = {"name": "cellxgene_census/dkd", "test_donors": ["control_3"]}

    manifest = op_v1.build_manifest(
        _args(classification_only=True),
        spec=spec,
        embedded=embedded,
    )

    assert manifest["dataset"] == "cellxgene_census/dkd"
    assert manifest["embedding"] == "cell_type PCA50"
    assert manifest["embedding_width"] == 3
    assert manifest["classification_only"] is True


def test_cli_main_delegates_to_v1_benchmark_runner(monkeypatch):
    calls = []
    args = Namespace(dataset="dkd")

    monkeypatch.setattr(cli, "run_openproblems_v1_benchmark", calls.append)

    cli.main(args)

    assert calls == [args]


def test_notebook_config_is_python_native_and_serializable():
    config = op_v1.OpenProblemsV1Config(
        dataset="dkd",
        input="input.h5ad",
        checkpoint="scprint-1.ckpt",
        output_dir="results",
    )

    assert config.as_dict()["dataset"] == "dkd"
    assert config.as_namespace().classification_only is False
