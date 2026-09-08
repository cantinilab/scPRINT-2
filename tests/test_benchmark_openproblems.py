from argparse import Namespace
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scprint2.benchmark import openproblems as op
from scripts import scprint2_op_full_no_assay as cli


def _args(**overrides):
    values = {
        "checkpoint": "checkpoint.ckpt",
        "classification_only": False,
        "scib_only": False,
        "mode": "zeroshot",
        "mmd_mode": "off",
        "mmd_weight": 0.03,
        "mmd_batch_key": "donor_id",
        "seed": 7,
    }
    values.update(overrides)
    return Namespace(**values)


def test_select_embedding_view_uses_full_or_cell_type_embedding():
    embedded = ad.AnnData(X=np.zeros((2, 1)))
    embedded.obsm["scprint_emb_cell_type_ontology_term_id"] = np.asarray(
        [[1.0, 2.0], [3.0, 4.0]], dtype=np.float64
    )
    selected = ["scprint_emb_other", "scprint_emb_disease_ontology_term_id"]

    name, blocks = op.select_embedding_view(
        embedded,
        selected,
        mode="zeroshot",
        embedding_view="full_no_assay",
    )
    assert name == "full_no_assay"
    assert blocks == selected

    name, blocks = op.select_embedding_view(
        embedded,
        selected,
        mode="zeroshot",
        embedding_view="cell_type",
    )
    assert name == "cell_type"
    assert blocks == ["scprint_emb_cell_type_ontology_term_id"]
    assert embedded.obsm["scprint_emb"].dtype == np.float32


def test_build_manifest_preserves_mmd_and_scib_flags():
    embedded = ad.AnnData(X=np.zeros((3, 1)))
    embedded.obsm["scprint_emb"] = np.zeros((3, 5), dtype=np.float32)
    spec = {"name": "cellxgene_census/dkd", "test_donors": ["control_3"]}

    manifest = op.build_manifest(
        _args(mode="finetune", mmd_mode="fixed", mmd_weight=2.5, scib_only=True),
        spec=spec,
        embedded=embedded,
        embedding_name="cell_type",
        embedding_blocks=["scprint_emb_cell_type_ontology_term_id"],
    )

    assert manifest["dataset"] == "cellxgene_census/dkd"
    assert manifest["embedding_width"] == 5
    assert manifest["mmd_mode"] == "fixed"
    assert manifest["mmd_weight"] == 2.5
    assert manifest["scib_only"] is True

    manifest = op.build_manifest(
        _args(mode="zeroshot", mmd_mode="fixed", mmd_weight=2.5),
        spec=spec,
        embedded=embedded,
        embedding_name="full_no_assay",
        embedding_blocks=["scprint_emb_other"],
    )

    assert manifest["mmd_mode"] == "not_applicable"
    assert manifest["mmd_weight"] == 0.0


def test_classification_scores_persists_logits_and_held_out_mask(monkeypatch):
    embedded = ad.AnnData(
        X=np.zeros((3, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["train", "test", "test"],
                "CL:1": [0.9, 0.1, 0.2],
                "CL:2": [0.1, 0.9, 0.8],
                "seurat_clusters": ["a", "b", "b"],
            },
            index=["cell-a", "cell-b", "cell-c"],
        ),
    )
    embedded.obsm["scprint_emb"] = np.ones((3, 2), dtype=np.float32)
    model = SimpleNamespace(label_decoders={}, labels_hierarchy={})

    def fake_compute(view, keys, *, label_decoders, labels_hierarchy):
        assert keys == [op.CELL_TYPE]
        return {op.CELL_TYPE: {"n_obs": view.n_obs}}

    def fake_refine(values, embedded_arg, return_raw=False):
        assert embedded_arg is embedded
        if return_raw:
            return values
        return values.argmax(1)

    monkeypatch.setattr(op, "compute_classification", fake_compute)
    monkeypatch.setattr(op, "zero_shot_annotation_with_refinement", fake_refine)

    scores = op.classification_scores(embedded, model, ["test"])

    assert scores["held_out_direct"] == {"n_obs": 2}
    assert scores["held_out_smooth"] == {"n_obs": 2}
    assert scores["held_out_cluster"] == {"n_obs": 2}
    assert embedded.obs["classification_held_out"].tolist() == [False, True, True]
    assert embedded.uns["classification_logit_labels"] == ["CL:1", "CL:2"]
    np.testing.assert_array_equal(
        embedded.obsm["classification_logits"],
        np.asarray([[0.9, 0.1], [0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
    )


def test_classification_scores_rejects_missing_held_out_donor():
    embedded = ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["train-a", "train-b"],
                "CL:1": [0.9, 0.8],
                "CL:2": [0.1, 0.2],
            }
        ),
    )
    embedded.obsm["scprint_emb"] = np.ones((2, 2), dtype=np.float32)
    model = SimpleNamespace(label_decoders={}, labels_hierarchy={})

    with pytest.raises(ValueError, match="No cells found"):
        op.classification_scores(embedded, model, ["held-out"])


def test_notebook_config_is_python_native_and_serializable():
    config = op.OpenProblemsConfig(
        dataset="dkd",
        mode="finetune",
        input="input.h5ad",
        checkpoint="small-v2.ckpt",
        output_dir="results",
        embedding_view="cell_type",
        mmd_mode="fixed",
        mmd_weight=0.03,
        mmd_batch_key="donor_id",
    )

    assert config.as_dict()["mmd_batch_key"] == "donor_id"
    assert config.as_namespace().dataset == "dkd"


def test_cli_main_delegates_to_benchmark_runner(monkeypatch):
    calls = []
    args = Namespace(dataset="dkd")

    monkeypatch.setattr(cli, "run_openproblems_benchmark", calls.append)

    cli.main(args)

    assert calls == [args]


def test_cli_enables_point_zero_three_mmd_by_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "scprint2_op_full_no_assay.py",
            "--dataset",
            "dkd",
            "--mode",
            "finetune",
            "--input",
            "input.h5ad",
            "--checkpoint",
            "small-v2.ckpt",
            "--output-dir",
            "results",
        ],
    )

    assert cli.parse_args().mmd_mode == "fixed"
    assert cli.parse_args().mmd_weight == 0.03
    assert cli.parse_args().mmd_batch_key == "donor_id"


def test_notebook_dataset_helpers_expose_held_out_split(tmp_path):
    source = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["train", "control_3", "control_3"],
                op.CELL_TYPE: ["CL:1", "CL:1", "CL:2"],
            },
            index=["a", "b", "c"],
        ),
    )
    input_path = tmp_path / "dkd.h5ad"
    source.write_h5ad(input_path)
    config = op.OpenProblemsConfig(
        dataset="dkd",
        mode="zeroshot",
        input=input_path,
        checkpoint="checkpoint.ckpt",
        output_dir=tmp_path,
    )

    loaded = op.load_openproblems_dataset(config)
    audit = op.summarize_openproblems_dataset(loaded, config)

    assert audit["training_cells"] == 1
    assert audit["held_out_cells"] == 2
    assert audit["held_out_donors"] == "control_3"


def test_finetune_helper_uses_user_selected_mmd_groups(monkeypatch):
    calls = []

    class FakeFinetuner:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __call__(self, *, model, train_data):
            calls.append({"train_cells": train_data.n_obs})
            return model

    monkeypatch.setattr(op, "FinetuneBatchClass", FakeFinetuner)
    adata = ad.AnnData(
        X=np.ones((3, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "donor_id": ["train", "control_3", "control_3"],
                "species": ["human", "human", "human"],
            }
        ),
    )
    config = op.OpenProblemsConfig(
        dataset="dkd",
        mode="finetune",
        input="input.h5ad",
        checkpoint="checkpoint.ckpt",
        output_dir="results",
        mmd_batch_key="species",
    )
    model = object()

    assert op.finetune_openproblems_model(model, adata, config) is model
    assert calls[0]["batch_key"] == "species"
    assert calls[0]["do_mmd_on"] == op.CELL_TYPE
    assert calls[0]["loss_scalers"]["mmd"] == 0.03
    assert calls[1]["train_cells"] == 1
