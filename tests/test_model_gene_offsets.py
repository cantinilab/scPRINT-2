from types import SimpleNamespace

import pandas as pd
import pytest

from scprint2.tasks._model_genes import (
    expected_model_gene_offsets,
    model_gene_dataframe,
    validate_collator_gene_offsets,
)


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
