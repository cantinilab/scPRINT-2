import anndata as ad
import numpy as np
import pandas as pd
import pytest

from op_scib import _align_solution


def _alignment_inputs():
    integrated = ad.AnnData(
        X=np.zeros((4, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["a", "a", "b", "b"],
                "cell_type": ["x", "y", "x", "y"],
                "nCount_RNA": [100, 205, 302, 410],
            },
            index=["cell-a", "cell-b", "cell-c", "cell-d"],
        ),
    )
    solution = ad.AnnData(
        X=np.zeros((4, 1)),
        obs=pd.DataFrame(
            {
                "donor_id": ["a", "a", "b", "b"],
                "cell_type": ["x", "y", "x", "y"],
                "size_factors": np.array([100, 200, 300, 400]) / 10_000,
            },
            index=["0", "1", "2", "3"],
        ),
    )
    return integrated, solution


def test_align_solution_validates_positional_identity():
    integrated, solution = _alignment_inputs()

    with pytest.warns(UserWarning, match="Aligned solution by position"):
        result, aligned = _align_solution(
            integrated,
            solution,
            batch_key="donor_id",
            label_key="cell_type",
        )

    assert aligned is not None
    assert aligned.obs_names.equals(result.obs_names)
    assert aligned.obs["donor_id"].tolist() == ["a", "a", "b", "b"]


def test_align_solution_rejects_positional_metadata_mismatch():
    integrated, solution = _alignment_inputs()
    solution.obs.iloc[1, solution.obs.columns.get_loc("cell_type")] = "x"

    with pytest.raises(ValueError, match="positional identity check failed"):
        _align_solution(
            integrated,
            solution,
            batch_key="donor_id",
            label_key="cell_type",
        )
