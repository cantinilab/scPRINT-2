import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scprint2.preprocessing.pandora import (
    PandoraBenchmarkConfig,
    _json_default,
    _valid_label_mask,
    attach_embedding,
    build_benchmark_data,
)
from scripts.benchmark_pandora_lung_scib import _parse_embedding_spec


def test_json_default_serializes_anndata_numpy_provenance() -> None:
    payload = {
        "cells": np.int64(3),
        "labels": np.array(["unknown", "mix"], dtype=object),
    }
    assert json.loads(json.dumps(payload, default=_json_default)) == {
        "cells": 3,
        "labels": ["unknown", "mix"],
    }


def test_valid_label_mask_is_case_insensitive() -> None:
    labels = pd.Series(["AT1", "unknown", " Mix ", "Macrophages"])
    mask = _valid_label_mask(labels, {"unknown", "mix"})
    assert mask.tolist() == [True, False, False, True]


def test_build_benchmark_data_filters_and_adds_pca() -> None:
    obs = pd.DataFrame(
        {
            "species": ["cat", "dog", "cat", "dog"],
            "cell_type_ontology_term_id": [
                "CL:0002062",
                "CL:0002062",
                "unknown",
                "CL:0002063",
            ],
        },
        index=[f"cell-{i}" for i in range(4)],
    )
    expression = ad.AnnData(
        X=sparse.csr_matrix(
            np.array(
                [[1, 0, 3, 0], [0, 2, 0, 4], [1, 1, 1, 1], [0, 5, 0, 1]],
                dtype=np.float32,
            )
        ),
        obs=obs,
    )
    embedded = ad.AnnData(obs=obs.copy())
    embedded.obsm["scprint_emb"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = build_benchmark_data(
        expression,
        embedded,
        "scprint_emb",
        "cell_type_ontology_term_id",
        {"unknown"},
        seed=42,
    )
    assert result.n_obs == 3
    assert result.obsm["scprint_emb"].shape == (3, 3)
    assert result.obsm["X_pca"].shape[0] == 3


def test_attach_embedding_supports_distinct_output_names() -> None:
    obs = pd.DataFrame(index=["a", "b", "c"])
    expression = ad.AnnData(X=np.ones((3, 3)), obs=obs.copy())
    embedded = ad.AnnData(obs=obs.copy())
    embedded.obsm["scprint_emb"] = np.ones((3, 2), dtype=np.float32)
    benchmark = expression[[True, False, True]].copy()
    alignment = attach_embedding(
        benchmark,
        expression,
        embedded,
        "scprint_emb",
        "scprint2_ft",
        np.array([True, False, True]),
    )
    assert benchmark.obsm["scprint2_ft"].shape == (2, 2)
    assert alignment == "exact_obs_names"


def test_attach_embedding_aligns_superset_by_cell_name() -> None:
    expression_obs = pd.DataFrame(index=["a", "b", "c"])
    expression = ad.AnnData(X=np.ones((3, 3)), obs=expression_obs)
    embedded = ad.AnnData(obs=pd.DataFrame(index=["extra", "c", "a", "b"]))
    embedded.obsm["model_emb"] = np.array(
        [[99, 99], [3, 30], [1, 10], [2, 20]], dtype=np.float32
    )
    benchmark = expression[[True, False, True]].copy()

    alignment = attach_embedding(
        benchmark,
        expression,
        embedded,
        "model_emb",
        "transcriptformer",
        np.array([True, False, True]),
    )

    assert benchmark.obsm["transcriptformer"].tolist() == [[1, 10], [3, 30]]
    assert alignment == "obs_names_subset"


def test_attach_embedding_recovers_transcriptformer_input_names() -> None:
    expression = ad.AnnData(
        X=np.ones((2, 2)), obs=pd.DataFrame(index=["original-a", "original-b"])
    )
    embedded = ad.AnnData(
        obs=pd.DataFrame(
            {
                "_transcriptformer_input_obs_name": [
                    "extra",
                    "original-b",
                    "original-a",
                ]
            },
            index=["generated-0", "generated-1", "generated-2"],
        )
    )
    embedded.obsm["model_emb"] = np.array(
        [[99, 99], [2, 20], [1, 10]], dtype=np.float32
    )
    benchmark = expression.copy()

    alignment = attach_embedding(
        benchmark,
        expression,
        embedded,
        "model_emb",
        "TranscriptFormer",
        np.array([True, True]),
    )

    assert benchmark.obsm["TranscriptFormer"].tolist() == [[1, 10], [2, 20]]
    assert alignment == "_transcriptformer_input_obs_name"


def test_attach_embedding_records_transcriptformer_feature_space_suffix() -> None:
    expression = ad.AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(
            index=["cell-a.mouse_homolog_ensmusg", "cell-b.mouse_homolog_ensmusg"]
        ),
    )
    embedded = ad.AnnData(
        obs=pd.DataFrame(
            {
                "_transcriptformer_input_obs_name": [
                    "cell-b.human_one2one",
                    "cell-a.human_one2one",
                ]
            },
            index=["generated-0", "generated-1"],
        )
    )
    embedded.obsm["model_emb"] = np.array([[2, 20], [1, 10]], dtype=np.float32)
    benchmark = expression.copy()

    alignment = attach_embedding(
        benchmark,
        expression,
        embedded,
        "model_emb",
        "TranscriptFormer",
        np.array([True, True]),
    )

    assert benchmark.obsm["TranscriptFormer"].tolist() == [[1, 10], [2, 20]]
    assert alignment.endswith(".human_one2one->.mouse_homolog_ensmusg")


def test_parse_embedding_spec() -> None:
    name, path, key = _parse_embedding_spec("transcriptformer=/tmp/tf.h5ad:model_emb")
    assert name == "transcriptformer"
    assert path == Path("/tmp/tf.h5ad")
    assert key == "model_emb"


def test_notebook_config_exposes_python_inputs() -> None:
    config = PandoraBenchmarkConfig(
        expression="expression.h5ad",
        embedding="embedding.h5ad",
        output="scores.tsv",
        embedding_name="scPRINT-2",
        additional_embeddings=(("TranscriptFormer", "tf.h5ad", "model_emb"),),
    )

    payload = config.as_dict()
    assert payload["embedding_name"] == "scPRINT-2"
    assert payload["additional_embeddings"][0]["key"] == "model_emb"
