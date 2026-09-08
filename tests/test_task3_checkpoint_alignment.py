import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scprint2.benchmark.task3 import (
    Task3Config,
    _align_to_checkpoint_genes,
    _classification_targets,
    _common_metadata,
    _ensure_historical_knn_graph,
    _finetune_settings,
    _mmd_class_token,
    _refuse_overwrite,
    _unknown_checkpoint_labels,
    _valid_label_mask,
    prepare_task3_dataset,
    summarize_task3_dataset,
    task3_output_paths,
)


def test_align_to_checkpoint_genes_orders_and_zero_pads():
    source = ad.AnnData(
        X=np.array([[3, 7], [5, 11]]),
        obs=pd.DataFrame(
            {"organism_ontology_term_id": ["NCBITaxon:9606"] * 2},
            index=["a", "b"],
        ),
        var=pd.DataFrame(index=["ENSG3", "ENSG1"]),
    )
    model = SimpleNamespace(_genes={"NCBITaxon:9606": ["ENSG1", "ENSG2", "ENSG3"]})

    aligned = _align_to_checkpoint_genes(source, model)

    assert aligned.var_names.tolist() == ["ENSG1", "ENSG2", "ENSG3"]
    assert aligned.X.toarray().tolist() == [[7, 0, 3], [11, 0, 5]]
    assert set(aligned.var["organism"]) == {"NCBITaxon:9606"}
    assert aligned.uns["checkpoint_gene_alignment"]["missing_genes_zero_filled"] == 1


def test_knn_graph_matches_scib_expression_preprocessing_and_survives_alignment():
    rng = np.random.default_rng(42)
    source = ad.AnnData(
        X=rng.poisson(2, size=(20, 6)).astype(np.float32),
        obs=pd.DataFrame(
            {"organism_ontology_term_id": ["NCBITaxon:10090"] * 20},
            index=[f"cell-{index}" for index in range(20)],
        ),
        var=pd.DataFrame(index=[f"ENSMUSG{index}" for index in range(6)]),
    )
    raw = source.X.copy()
    model = SimpleNamespace(
        _genes={
            "NCBITaxon:10090": [
                "ENSMUSG2",
                "ENSMUSG0",
                "ENSMUSG_missing",
                "ENSMUSG5",
            ]
        }
    )

    rebuilt = _ensure_historical_knn_graph(source, seed=42)
    aligned = _align_to_checkpoint_genes(source, model)

    assert rebuilt is True
    np.testing.assert_array_equal(source.X, raw)
    assert source.obsm["X_pca"].shape == (20, 5)
    assert source.uns["scprint_knn_cells"]["normalization"] == (
        "normalize_total_target_sum_10000_then_log1p"
    )
    assert source.uns["scprint_knn_cells"]["use_highly_variable"] is False
    assert aligned.obsp["connectivities"].shape == (20, 20)
    assert aligned.obsp["distances"].shape == (20, 20)
    np.testing.assert_array_equal(aligned.obsm["X_pca"], source.obsm["X_pca"])
    assert "log1p" not in source.uns


def test_knn_graph_can_be_forced_to_rebuild_existing_neighbors():
    rng = np.random.default_rng(7)
    source = ad.AnnData(
        X=rng.poisson(2, size=(20, 6)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(20)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(6)]),
    )
    source.obsp["connectivities"] = np.eye(20, dtype=np.float32)
    source.obsp["distances"] = np.eye(20, dtype=np.float32)
    source.uns["neighbors"] = {"params": {"use_rep": "stale"}}
    stale = source.obsp["connectivities"].copy()

    rebuilt = _ensure_historical_knn_graph(source, seed=42, force_rebuild=True)

    assert rebuilt is True
    assert source.uns["scprint_knn_cells"]["forced_rebuild"] is True
    assert source.uns["neighbors"]["params"]["use_rep"] == "X_pca"
    assert not np.array_equal(source.obsp["connectivities"].toarray(), stale)


def test_valid_label_mask_excludes_unknown_training_labels():
    labels = pd.Series(["CL:0002062", "unknown", " Unknown ", ""])
    mask = _valid_label_mask(labels, {"unknown"})
    assert mask.tolist() == [True, False, False, False]


def test_unknown_checkpoint_labels_reports_terms_that_would_be_coerced():
    model = SimpleNamespace(
        label_decoders={
            "cell_type_ontology_term_id": {
                0: "unknown",
                1: "CL:0000327",
                2: "CL:0002320",
            }
        }
    )
    labels = pd.Series(["CL:0000327", "CL:7770002", "CL:0002320"])

    assert _unknown_checkpoint_labels(
        model, labels, "cell_type_ontology_term_id"
    ) == ["CL:7770002"]


def test_species_mmd_uses_cell_type_token_and_records_physical_filter(tmp_path: Path):
    args = SimpleNamespace(
        mode="scprint2-ft",
        notebook_provenance="notebook.ipynb",
        input=str(tmp_path / "filtered.h5ad"),
        output=str(tmp_path / "embedding.h5ad"),
        checkpoint=str(tmp_path / "small-v2.ckpt"),
        seed=42,
        batch_key="species",
        cell_type_key="cell_type_ontology_term_id",
        batch_size=8,
        num_workers=2,
        embed_how="random expr",
        embed_max_len=2200,
        use_knn=False,
        legacy_detached_mmd=False,
        restore_best_model=True,
        align_checkpoint_genes=True,
        finetune_max_len=2200,
        num_epochs=8,
        mmd_target="species",
        mmd_scale=0.03,
        finetune_objective="reconstruction-mmd",
        include_organism_classification=False,
    )
    adata = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.uns["pandora_cell_filter"] = {
        "source_cells": np.int64(4),
        "filtered_cells": np.int64(3),
        "excluded_cells": 1,
        "excluded_labels": np.array(["unknown", "mix"], dtype=object),
        "physical_filter": True,
    }

    assert _mmd_class_token(args) == "cell_type_ontology_term_id"

    _common_metadata(args, "scprint_emb", adata=adata)

    metadata = json.loads((tmp_path / "embedding.metadata.json").read_text())
    assert metadata["settings"]["finetune"]["mmd_target"] == "species"
    assert metadata["settings"]["finetune"]["mmd_batch_key"] == "species"
    assert metadata["settings"]["finetune"]["do_mmd_on"] == (
        "cell_type_ontology_term_id"
    )
    assert metadata["settings"]["finetune"]["loss_scalers"]["mmd"] == 0.03
    assert metadata["settings"]["finetune"]["classification_targets"] == [
        "cell_type_ontology_term_id"
    ]
    assert metadata["settings"]["finetune"]["organism_classification"] is False
    assert metadata["settings"]["finetune"]["organism_decoder_trainable"] is False
    assert metadata["settings"]["finetune"]["knn_forward_enabled"] is False
    assert metadata["physical_filter"]["filtered_cells"] == 3
    assert metadata["physical_filter"]["excluded_labels"] == ["unknown", "mix"]


def test_cell_type_only_finetune_disables_all_non_cell_type_objectives():
    args = SimpleNamespace(
        finetune_objective="cell-type-only",
        cell_type_key="cell_type_ontology_term_id",
    )

    settings = _finetune_settings(args)

    assert settings["do_mmd_on"] is None
    assert settings["learn_batches_on"] is None
    assert settings["loss_scalers"] == {
        "expr": 0,
        "class": 1,
        "mmd": 0,
        "kl": 0,
        "organism_ontology_term_id": 0,
    }


def test_reconstruction_mmd_uses_configured_scale():
    args = SimpleNamespace(
        finetune_objective="reconstruction-mmd",
        cell_type_key="cell_type_ontology_term_id",
        mmd_scale=0.03,
        include_organism_classification=False,
    )

    settings = _finetune_settings(args)

    assert settings["do_mmd_on"] == "cell_type_ontology_term_id"
    assert settings["loss_scalers"]["mmd"] == 0.03
    assert settings["loss_scalers"]["organism_ontology_term_id"] == 0


def test_task3_compatible_finetune_includes_organism_classification_explicitly():
    args = SimpleNamespace(
        finetune_objective="reconstruction-mmd",
        cell_type_key="cell_type_ontology_term_id",
        mmd_scale=0.03,
        include_organism_classification=True,
    )

    settings = _finetune_settings(args)

    assert _classification_targets(args) == [
        "cell_type_ontology_term_id",
        "organism_ontology_term_id",
    ]
    assert settings["loss_scalers"]["organism_ontology_term_id"] == 1


def test_task3_notebook_config_has_explicit_python_fields(tmp_path: Path):
    config = Task3Config(
        mode="scprint2-ft",
        input=tmp_path / "task3.h5ad",
        checkpoint=tmp_path / "small-v2.ckpt",
        output=tmp_path / "embedding.h5ad",
        notebook_provenance="task3 notebook",
        batch_key="orig.ident",
        mmd_target="species",
        mmd_scale=0.03,
        use_knn=True,
    )

    assert config.as_dict()["batch_key"] == "orig.ident"
    assert (
        Task3Config(
            mode="scprint2-ft",
            input=tmp_path / "task3.h5ad",
            checkpoint=tmp_path / "small-v2.ckpt",
            output=tmp_path / "default_embedding.h5ad",
            notebook_provenance="task3 notebook",
        ).mmd_scale
        == 0.03
    )
    assert config.as_namespace().notebook_provenance == "task3 notebook"


def test_task3_refuses_existing_sibling_artifact_before_run(tmp_path: Path):
    output = tmp_path / "embedding.h5ad"
    metadata = output.with_suffix(".metadata.json")
    checkpoint = output.with_suffix(".ckpt")
    metadata.write_text("existing\n")

    try:
        _refuse_overwrite(output, metadata, checkpoint)
    except FileExistsError as error:
        assert str(metadata) in str(error)
    else:
        raise AssertionError("existing sibling provenance must block a fresh run")

    assert not output.exists()
    assert not checkpoint.exists()


def test_task3_notebook_helpers_expose_data_and_output_contract(tmp_path: Path):
    config = Task3Config(
        mode="scprint2-ft",
        input=tmp_path / "task3.h5ad",
        checkpoint=tmp_path / "small-v2.ckpt",
        output=tmp_path / "embedding.h5ad",
        notebook_provenance="notebook",
        batch_key="species",
    )
    adata = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "species": ["cat", "cat", "tiger"],
                "cell_type_ontology_term_id": ["CL:1", "unknown", "CL:2"],
            },
            index=["a", "b", "c"],
        ),
    )

    audit = summarize_task3_dataset(adata, config)

    assert audit["cells"] == 3
    assert audit["batches"] == 2
    assert audit["unknown_labels"] == 1
    assert task3_output_paths(config) == [
        tmp_path / "embedding.h5ad",
        tmp_path / "embedding.metadata.json",
        tmp_path / "embedding.ckpt",
    ]


def test_prepare_task3_dataset_returns_knn_decision_without_hiding_it():
    source = ad.AnnData(
        X=np.ones((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
    )
    model = SimpleNamespace(expr_emb_style="single-cell")
    config = Task3Config(
        mode="scprint2-zero-shot",
        input="input.h5ad",
        checkpoint="checkpoint.ckpt",
        output="output.h5ad",
        notebook_provenance="notebook",
        use_knn=True,
    )

    prepared, rebuilt = prepare_task3_dataset(source, model, config)

    assert prepared is source
    assert rebuilt is False
