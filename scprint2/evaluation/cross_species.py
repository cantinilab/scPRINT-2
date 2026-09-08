"""Reusable cross-species embedding evaluation helpers."""

from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

TOKEN_KEYS = {
    "cell_type": "scprint_emb_cell_type_ontology_term_id",
    "disease": "scprint_emb_disease_ontology_term_id",
    "age": "scprint_emb_age_group",
    "tissue": "scprint_emb_tissue_ontology_term_id",
}
EXCLUDED_ALL_TOKEN_KEYS = {
    "scprint_emb_assay_ontology_term_id",
    "scprint_emb_organism_ontology_term_id",
}
ORGANISM_TOKEN_KEY = "scprint_emb_organism_ontology_term_id"


def build_expression_baselines(
    source: ad.AnnData, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, dict]:
    prepared = source.copy()
    for key in ("connectivities", "distances"):
        prepared.obsp.pop(key, None)
    for key in ("neighbors", "pca"):
        prepared.uns.pop(key, None)
    prepared.uns.pop("log1p", None)
    prepared.obsm.pop("X_pca", None)
    prepared.varm.pop("PCs", None)

    sc.pp.normalize_total(prepared, target_sum=10_000)
    sc.pp.log1p(prepared)
    sc.pp.pca(prepared)
    pca = np.asarray(prepared.obsm["X_pca"], dtype=np.float32)
    random = np.random.default_rng(seed).random(pca.shape).astype(np.float32)
    metadata = {
        "normalization": "normalize_total_target_sum_10000_then_log1p",
        "pca_implementation": "scanpy.pp.pca_default",
        "pca_dimension": int(pca.shape[1]),
        "random_distribution": "uniform_0_1",
        "seed": seed,
    }
    return pca, random, metadata


def all_token_keys_except_assay_and_organism(
    source: ad.AnnData,
) -> dict[str, str]:
    selected = sorted(
        key
        for key in source.obsm
        if key.startswith("scprint_emb_") and key not in EXCLUDED_ALL_TOKEN_KEYS
    )
    if not selected:
        raise KeyError("No scPRINT token embeddings are available for concatenation")
    return {key.removeprefix("scprint_emb_"): key for key in selected}


def all_token_keys_except_organism(source: ad.AnnData) -> dict[str, str]:
    """Select every persisted scPRINT token embedding except organism."""
    selected = sorted(
        key
        for key in source.obsm
        if key.startswith("scprint_emb_") and key != ORGANISM_TOKEN_KEY
    )
    if not selected:
        raise KeyError("No scPRINT token embeddings are available for concatenation")
    return {key.removeprefix("scprint_emb_"): key for key in selected}


def build_token_concat_pca(
    source: ad.AnnData,
    seed: int = 42,
    token_keys: dict[str, str] | None = None,
) -> tuple[np.ndarray, dict]:
    token_keys = TOKEN_KEYS if token_keys is None else token_keys
    missing = [name for name, key in token_keys.items() if key not in source.obsm]
    if missing:
        raise KeyError(f"Required token embeddings are missing: {missing}")

    blocks = [
        np.asarray(source.obsm[key], dtype=np.float32) for key in token_keys.values()
    ]
    concatenated = np.concatenate(blocks, axis=1)
    n_components = min(50, concatenated.shape[0] - 1, concatenated.shape[1])
    if n_components < 2:
        raise RuntimeError("Token-concat PCA requires at least three cells/features")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    embedding = pca.fit_transform(concatenated).astype(np.float32, copy=False)
    metadata = {
        "selected_tokens": list(token_keys),
        "selected_obsm_keys": list(token_keys.values()),
        "input_dimensions": [int(block.shape[1]) for block in blocks],
        "concatenated_dimension": int(concatenated.shape[1]),
        "pca_dimension": n_components,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "seed": seed,
    }
    return embedding, metadata
