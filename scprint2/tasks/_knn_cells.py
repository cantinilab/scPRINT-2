"""Reliable Scanpy-neighbor extraction for metacell expression encoders."""

from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData
from scdataloader.data import SimpleAnnDataset
from scipy import sparse


class ScanpyNeighborAnnDataset(SimpleAnnDataset):
    """Expose the six actual nearest cells from ``obsp['distances']``.

    ``scdataloader==2.1.3`` densifies the sparse Scanpy distance row and ranks
    ``-1 / (distance - 1e-6)``.  Structural zeros then outrank every stored
    edge, so the returned cells are generally non-neighbors. This wrapper
    ranks only stored Scanpy edges and keeps the public batch fields expected
    by scPRINT's metacell encoder.
    """

    def __init__(
        self,
        adata: AnnData,
        obs_to_output: Optional[list[str]] = None,
        layer: Optional[str] = None,
        get_knn_cells: bool = False,
        encoder: Optional[dict[str, dict]] = None,
        n_neighbors: int = 6,
    ) -> None:
        super().__init__(
            adata,
            obs_to_output=obs_to_output or [],
            layer=layer,
            # Never invoke scdataloader's sparse-zero ranking implementation.
            get_knn_cells=False,
            encoder=encoder,
        )
        self.get_knn_cells = get_knn_cells
        self.n_neighbors = n_neighbors
        self.knn_indices: np.ndarray | None = None
        self.knn_distances: np.ndarray | None = None
        if not get_knn_cells:
            return
        if "distances" not in adata.obsp:
            raise ValueError("Scanpy neighbors require adata.obsp['distances']")
        if adata.n_obs <= n_neighbors:
            raise ValueError(
                f"PCA neighbors require more than {n_neighbors} cells; "
                f"received {adata.n_obs}"
            )

        graph = adata.obsp["distances"]
        graph = graph.tocsr() if sparse.issparse(graph) else sparse.csr_matrix(graph)
        selected_indices = np.empty((adata.n_obs, n_neighbors), dtype=np.int64)
        selected_distances = np.empty((adata.n_obs, n_neighbors), dtype=np.float32)
        for cell_index in range(adata.n_obs):
            start, stop = graph.indptr[cell_index : cell_index + 2]
            row_indices = graph.indices[start:stop]
            row_distances = graph.data[start:stop]
            not_self = row_indices != cell_index
            row_indices = row_indices[not_self]
            row_distances = row_distances[not_self]
            if row_indices.size < n_neighbors:
                raise ValueError(
                    f"cell {cell_index} has only {row_indices.size} stored neighbors; "
                    "rebuild the Scanpy graph on this dataset split"
                )
            order = np.argsort(row_distances, kind="stable")[:n_neighbors]
            selected_indices[cell_index] = row_indices[order]
            selected_distances[cell_index] = row_distances[order]
        self.knn_indices = selected_indices
        self.knn_distances = selected_distances

    def __getitem__(self, idx: int) -> dict:
        out = {"X": self.adataX[idx].reshape(-1)}
        for name, value in self.obs_to_output.iloc[idx].items():
            out[name] = self.encoder[name][value] if name in self.encoder else value
        if self.get_knn_cells:
            assert self.knn_indices is not None
            assert self.knn_distances is not None
            out["knn_cells"] = np.asarray(self.adataX[self.knn_indices[idx]], dtype=int)
            out["knn_cells_info"] = self.knn_distances[idx].copy()
        return out
