from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scdataloader import Collator

from scprint2.tasks import (
    expected_model_gene_offsets,
    model_gene_dataframe,
    validate_collator_gene_offsets,
)
from scprint2.tasks._model_genes import collator_for_organism_blocks


def _model():
    return SimpleNamespace(
        organisms=["mouse", "rat", "human"],
        _genes={
            "mouse": ["m1", "m2"],
            "rat": ["r1", "r2", "r3"],
            "human": ["h1", "h2"],
        },
    )


def test_model_gene_dataframe_preserves_multispecies_checkpoint_offsets():
    model = _model()
    input_var = pd.DataFrame(
        {"organism": pd.Categorical(["human", "human"])},
        index=["h1", "h2"],
    )

    genedf = model_gene_dataframe(model, input_var)

    assert genedf.index.tolist() == ["m1", "m2", "r1", "r2", "r3", "h1", "h2"]
    assert expected_model_gene_offsets(model) == {"mouse": 0, "rat": 2, "human": 5}


def test_offset_guard_rejects_single_species_reindexing():
    model = _model()
    broken_collator = SimpleNamespace(start_idx={"human": 0})

    with pytest.raises(RuntimeError, match="human starts at 0, expected 5"):
        validate_collator_gene_offsets(broken_collator, model, ["human"])


def test_model_gene_dataframe_rejects_missing_or_reordered_checkpoint_genes():
    model = _model()
    reordered = pd.DataFrame(
        {"organism": ["human", "human"]}, index=["h2", "h1"]
    )

    with pytest.raises(RuntimeError, match="not in checkpoint order"):
        model_gene_dataframe(model, reordered)


def test_offset_guard_accepts_checkpoint_global_indexing():
    model = _model()
    collator = SimpleNamespace(start_idx={"human": 5})

    validate_collator_gene_offsets(collator, model, ["human"])


def test_multiorganism_collator_slices_expression_and_knn_blocks():
    class CapturingCollator:
        organism_name = "organism_ontology_term_id"
        accepted_genes = {
            0: np.array([True, True]),
            1: np.array([True, True, True]),
        }

        def __call__(self, batch):
            return batch

    var = pd.DataFrame(
        {"organism": ["mouse", "mouse", "human", "human", "human"]},
        index=["m1", "m2", "h1", "h2", "h3"],
    )
    wrapper = collator_for_organism_blocks(
        CapturingCollator(),
        var,
        ["mouse", "human"],
        {"mouse": 0, "human": 1},
    )
    batch = [
        {
            "organism_ontology_term_id": 0,
            "X": np.array([1, 2, 0, 0, 0]),
            "knn_cells": np.array([[3, 4, 0, 0, 0]]),
        },
        {
            "organism_ontology_term_id": 1,
            "X": np.array([0, 0, 5, 6, 7]),
            "knn_cells": np.array([[0, 0, 8, 9, 10]]),
        },
    ]

    result = wrapper(batch)

    assert result[0]["X"].tolist() == [1, 2]
    assert result[0]["knn_cells"].tolist() == [[3, 4]]
    assert result[1]["X"].tolist() == [5, 6, 7]
    assert result[1]["knn_cells"].tolist() == [[8, 9, 10]]


def test_multiorganism_wrapper_runs_real_scdataloader_collator():
    var = pd.DataFrame(
        {"organism": ["mouse", "mouse", "human", "human", "human"]},
        index=["m1", "m2", "h1", "h2", "h3"],
    )
    collator = Collator(
        organisms=["mouse", "human"],
        org_to_id={"mouse": 0, "human": 1},
        valid_genes=list(var.index),
        genedf=var,
        class_names=[],
    )
    collator.organism_ids = {0, 1}
    wrapper = collator_for_organism_blocks(
        collator,
        var,
        ["mouse", "human"],
        {"mouse": 0, "human": 1},
    )

    mouse = wrapper(
        [{"organism_ontology_term_id": 0, "X": np.array([1, 2, 0, 0, 0])}]
    )
    human = wrapper(
        [{"organism_ontology_term_id": 1, "X": np.array([0, 0, 3, 4, 5])}]
    )

    assert mouse["x"].tolist() == [[1.0, 2.0]]
    assert mouse["genes"].tolist() == [[0, 1]]
    assert human["x"].tolist() == [[3.0, 4.0, 5.0]]
    assert human["genes"].tolist() == [[2, 3, 4]]
