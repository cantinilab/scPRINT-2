import anndata as ad
import numpy as np
from scipy import sparse

from scprint2.tasks import ScanpyNeighborAnnDataset


def test_pca_neighbor_dataset_returns_actual_nearest_cells():
    expression = np.arange(8, dtype=np.int64)[:, None]
    adata = ad.AnnData(X=expression)
    positions = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0])
    distances = np.abs(positions[:, None] - positions[None, :]).astype(np.float32)
    adata.obsp["distances"] = sparse.csr_matrix(distances)

    dataset = ScanpyNeighborAnnDataset(
        adata,
        get_knn_cells=True,
        n_neighbors=2,
    )

    first = dataset[0]
    np.testing.assert_array_equal(first["knn_cells"].ravel(), [1, 2])
    np.testing.assert_allclose(first["knn_cells_info"], [1.0, 2.0])
    assert 0 not in dataset.knn_indices[0]


def test_pca_neighbor_dataset_requires_scanpy_graph():
    adata = ad.AnnData(X=np.ones((8, 2), dtype=np.int64))

    try:
        ScanpyNeighborAnnDataset(adata, get_knn_cells=True)
    except ValueError as error:
        assert "distances" in str(error)
    else:
        raise AssertionError("missing Scanpy graph should fail")


def test_pca_neighbor_dataset_ignores_implicit_sparse_zeros():
    expression = np.arange(8, dtype=np.int64)[:, None]
    adata = ad.AnnData(X=expression)
    rows = []
    columns = []
    values = []
    for cell_index in range(8):
        rows.extend([cell_index, cell_index])
        columns.extend([(cell_index + 1) % 8, (cell_index + 2) % 8])
        values.extend([0.1, 0.2])
    adata.obsp["distances"] = sparse.csr_matrix(
        (values, (rows, columns)), shape=(8, 8), dtype=np.float32
    )

    dataset = ScanpyNeighborAnnDataset(
        adata,
        get_knn_cells=True,
        n_neighbors=2,
    )

    np.testing.assert_array_equal(dataset.knn_indices[0], [1, 2])
    np.testing.assert_array_equal(dataset[0]["knn_cells"].ravel(), [1, 2])
