import anndata as ad
import numpy as np
import pandas as pd

from scprint2.evaluation.cross_species import build_expression_baselines


def test_expression_baselines_are_pca_from_counts_and_seeded_random():
    rng = np.random.default_rng(7)
    source = ad.AnnData(
        X=rng.poisson(2, size=(24, 8)).astype(np.float32),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(24)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(8)]),
    )
    raw = source.X.copy()

    pca_a, random_a, metadata = build_expression_baselines(source, seed=42)
    pca_b, random_b, _ = build_expression_baselines(source, seed=42)

    np.testing.assert_array_equal(source.X, raw)
    np.testing.assert_allclose(pca_a, pca_b)
    np.testing.assert_array_equal(random_a, random_b)
    assert pca_a.shape == random_a.shape == (24, 7)
    assert metadata["normalization"] == (
        "normalize_total_target_sum_10000_then_log1p"
    )
    assert metadata["pca_dimension"] == 7
