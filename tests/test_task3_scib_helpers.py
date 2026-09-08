from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scprint2.evaluation import task3_scib
from scripts.benchmark_task3_scib113_precomputed import (
    _parse_embedding,
    _parse_obs_name_replace,
)
from scripts.prepare_task3_paper_scib import prepare


def test_parse_embedding_specs_keep_cli_errors() -> None:
    assert task3_scib.parse_embedding_spec("m=/tmp/result.h5ad:X") == (
        "m",
        task3_scib.Path("/tmp/result.h5ad"),
        "X",
    )
    with pytest.raises(ValueError, match="expected METHOD=PATH:OBSM_KEY"):
        task3_scib.parse_embedding_spec("bad")
    with pytest.raises(Exception, match="embedding must be METHOD=PATH:OBSM_KEY"):
        _parse_embedding("bad")


def test_parse_obs_name_replace_keeps_cli_errors() -> None:
    assert task3_scib.parse_obs_name_replace_spec("-source=-embedding") == (
        "-source",
        "-embedding",
    )
    with pytest.raises(ValueError, match="SOURCE_SUFFIX=EMBEDDING_SUFFIX"):
        task3_scib.parse_obs_name_replace_spec("bad")
    with pytest.raises(Exception, match="SOURCE_SUFFIX=EMBEDDING_SUFFIX"):
        _parse_obs_name_replace("bad")


def test_load_aligned_embedding_supports_suffix_mapping(tmp_path) -> None:
    source = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["cell-a-source", "cell-b-source"]),
    )
    result = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["cell-b-embedding", "cell-a-embedding"]),
    )
    result.obsm["model_emb"] = np.array([[2, 20], [1, 10]], dtype=np.float32)
    path = tmp_path / "embedding.h5ad"
    result.write_h5ad(path)

    values = task3_scib.load_aligned_embedding(
        source,
        path,
        "model_emb",
        obs_name_replace=("-source", "-embedding"),
    )

    assert values.tolist() == [[1, 10], [2, 20]]


def test_reference_pseudotime_preserves_disconnected_cells_as_missing(
    monkeypatch,
) -> None:
    prepared = ad.AnnData(
        X=np.ones((3, 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"species": pd.Categorical(["mouse", "mouse", "mouse"])},
            index=["m1", "m2", "m3"],
        ),
    )

    def highly_variable_genes(data):
        data.var["highly_variable"] = True

    monkeypatch.setattr(task3_scib.sc.pp, "highly_variable_genes", highly_variable_genes)
    monkeypatch.setattr(task3_scib.sc.tl, "pca", lambda *args, **kwargs: None)
    monkeypatch.setattr(task3_scib.sc.pp, "neighbors", lambda *args, **kwargs: None)
    monkeypatch.setattr(task3_scib.sc.tl, "diffmap", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        task3_scib.sc.tl,
        "dpt",
        lambda data: data.obs.__setitem__("dpt_pseudotime", [0.0, np.nan, 1.0]),
    )

    audit = task3_scib.precompute_reference_pseudotime(prepared, "species")

    assert prepared.obs["dpt_pseudotime"].isna().tolist() == [False, True, False]
    assert audit["mouse"]["finite_pseudotime_cells"] == 2
    assert audit["mouse"]["missing_pseudotime_cells"] == 1


def test_prepare_paper_scib_writes_ordered_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        task3_scib,
        "assert_task3_scib113_environment",
        lambda mismatch_message="": {"scib": "1.1.3", "scanpy": "1.9.1"},
    )
    source = ad.AnnData(
        X=sp.csr_matrix(np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.float32)),
        obs=pd.DataFrame(
            {
                "orig.ident": ["mouse", "human", "mouse", "human"],
                "celltype": ["a", "b", "c", "d"],
            },
            index=["m1", "h1", "m2", "h2"],
        ),
    )
    embedding = ad.AnnData(
        X=np.ones((4, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["h2", "m2", "h1", "m1"]),
    )
    embedding.obsm["model_emb"] = np.array(
        [[40, 400], [30, 300], [20, 200], [10, 100]], dtype=np.float32
    )
    source_path = tmp_path / "source.h5ad"
    embedding_path = tmp_path / "embedding.h5ad"
    paper_root = tmp_path / "paper"
    source.write_h5ad(source_path)
    embedding.write_h5ad(embedding_path)

    prepare(
        SimpleNamespace(
            source=str(source_path),
            embedding=str(embedding_path),
            embedding_key="model_emb",
            paper_root=str(paper_root),
            batch_key="orig.ident",
            label_key="celltype",
        )
    )

    table = pd.read_csv(
        paper_root / "output" / "method_outputs" / "saturn" / "task3_saturn_embedding.txt",
        index_col=0,
    )
    assert table.index.tolist() == ["h1", "h2", "m1", "m2"]
    assert table[["dim_0", "dim_1"]].to_numpy().tolist() == [
        [20, 200],
        [40, 400],
        [10, 100],
        [30, 300],
    ]
    metadata = pd.read_json(paper_root / "run_metadata.json", typ="series")
    assert metadata["batch_order"] == ["human", "mouse"]
