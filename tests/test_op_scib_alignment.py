import anndata as ad
import numpy as np
import pandas as pd
import pytest

from op_scib import (
    _align_solution,
    _scanpy_distances_as_neighbors,
    select_op_datasets,
)


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


def test_align_solution_accepts_label_drift_and_uses_reference_labels():
    integrated, solution = _alignment_inputs()
    solution.obs.iloc[1, solution.obs.columns.get_loc("cell_type")] = "x"

    with pytest.warns(UserWarning, match="labels are taken from"):
        result, _ = _align_solution(
            integrated,
            solution,
            batch_key="donor_id",
            label_key="cell_type",
        )

    assert result.obs["cell_type"].tolist() == ["x", "x", "x", "y"]


def test_align_solution_rejects_positional_batch_mismatch():
    integrated, solution = _alignment_inputs()
    solution.obs.iloc[1, solution.obs.columns.get_loc("donor_id")] = "b"

    with pytest.raises(ValueError, match="positional identity check failed"):
        _align_solution(
            integrated,
            solution,
            batch_key="donor_id",
            label_key="cell_type",
        )


def test_align_solution_rejects_library_size_reordering_within_batch():
    integrated, solution = _alignment_inputs()
    solution.obs["size_factors"] = solution.obs["size_factors"].iloc[::-1].to_numpy()

    with pytest.raises(ValueError, match="library-size fingerprints"):
        _align_solution(
            integrated,
            solution,
            batch_key="donor_id",
            label_key="cell_type",
        )


def test_scanpy_distances_as_neighbors_adds_self_edges():
    adata = ad.AnnData(X=np.zeros((3, 1)))
    adata.obsp["distances"] = np.array(
        [
            [0.0, 0.2, 0.4],
            [0.2, 0.0, 0.3],
            [0.4, 0.3, 0.0],
        ]
    )

    neighbors = _scanpy_distances_as_neighbors(adata)

    assert neighbors.indices.shape == (3, 3)
    assert neighbors.indices[:, 0].tolist() == [0, 1, 2]
    assert neighbors.distances[:, 0].tolist() == [0.0, 0.0, 0.0]


def test_select_op_datasets_accepts_slugs(monkeypatch):
    monkeypatch.setenv("OP_DATASETS", "hypomap,mouse_pancreas_atlas")
    datasets = {
        "cellxgene_census/dkd": 1,
        "cellxgene_census/hypomap": 2,
        "cellxgene_census/mouse_pancreas_atlas": 3,
    }

    assert select_op_datasets(datasets) == {
        "cellxgene_census/hypomap": 2,
        "cellxgene_census/mouse_pancreas_atlas": 3,
    }
