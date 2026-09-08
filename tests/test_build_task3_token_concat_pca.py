import anndata as ad
import numpy as np
import pytest

from scprint2.evaluation.cross_species import (
    EXCLUDED_ALL_TOKEN_KEYS,
    ORGANISM_TOKEN_KEY,
    TOKEN_KEYS,
    all_token_keys_except_assay_and_organism,
    all_token_keys_except_organism,
    build_token_concat_pca,
)


def test_build_token_concat_pca_uses_requested_four_tokens() -> None:
    source = ad.AnnData(X=np.ones((8, 3), dtype=np.float32))
    widths = [4, 3, 2, 3]
    for (name, key), width in zip(TOKEN_KEYS.items(), widths, strict=True):
        source.obsm[key] = np.arange(8 * width, dtype=np.float32).reshape(8, width)

    embedding, metadata = build_token_concat_pca(source, seed=42)

    assert list(TOKEN_KEYS) == ["cell_type", "disease", "age", "tissue"]
    assert embedding.shape == (8, 7)
    assert metadata["input_dimensions"] == widths
    assert metadata["concatenated_dimension"] == sum(widths)
    assert metadata["pca_dimension"] == 7


def test_build_token_concat_pca_requires_every_requested_token() -> None:
    source = ad.AnnData(X=np.ones((4, 3), dtype=np.float32))
    source.obsm[TOKEN_KEYS["cell_type"]] = np.ones((4, 2), dtype=np.float32)

    with pytest.raises(KeyError, match="disease"):
        build_token_concat_pca(source)


def test_all_token_policy_excludes_only_assay_and_organism_tokens() -> None:
    source = ad.AnnData(X=np.ones((8, 3), dtype=np.float32))
    keys = [
        "scprint_emb_cell_type_ontology_term_id",
        "scprint_emb_sex_ontology_term_id",
        "scprint_emb_cell_culture",
        "scprint_emb_other",
        *sorted(EXCLUDED_ALL_TOKEN_KEYS),
    ]
    for key in keys:
        source.obsm[key] = np.ones((8, 2), dtype=np.float32)
    source.obsm["scprint_emb"] = np.ones((8, 2), dtype=np.float32)
    source.obsm["X_pca"] = np.ones((8, 2), dtype=np.float32)

    selected = all_token_keys_except_assay_and_organism(source)
    embedding, metadata = build_token_concat_pca(
        source, seed=42, token_keys=selected
    )

    assert set(selected.values()).isdisjoint(EXCLUDED_ALL_TOKEN_KEYS)
    assert set(selected.values()) == {
        "scprint_emb_cell_type_ontology_term_id",
        "scprint_emb_sex_ontology_term_id",
        "scprint_emb_cell_culture",
        "scprint_emb_other",
    }
    assert embedding.shape == (8, 7)
    assert metadata["concatenated_dimension"] == 8


def test_all_except_organism_policy_keeps_assay_and_other_tokens() -> None:
    source = ad.AnnData(X=np.ones((8, 3), dtype=np.float32))
    keys = [
        "scprint_emb_cell_type_ontology_term_id",
        "scprint_emb_assay_ontology_term_id",
        "scprint_emb_sex_ontology_term_id",
        "scprint_emb_other",
        ORGANISM_TOKEN_KEY,
    ]
    for key in keys:
        source.obsm[key] = np.ones((8, 2), dtype=np.float32)
    source.obsm["scprint_emb"] = np.ones((8, 2), dtype=np.float32)

    selected = all_token_keys_except_organism(source)
    embedding, metadata = build_token_concat_pca(
        source, seed=42, token_keys=selected
    )

    assert ORGANISM_TOKEN_KEY not in selected.values()
    assert set(selected.values()) == {
        "scprint_emb_cell_type_ontology_term_id",
        "scprint_emb_assay_ontology_term_id",
        "scprint_emb_sex_ontology_term_id",
        "scprint_emb_other",
    }
    assert embedding.shape == (8, 7)
    assert metadata["concatenated_dimension"] == 8
