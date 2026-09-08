from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd

from scprint2.evaluation import openproblems_scib as op_scib
from scripts import score_openproblems_embedding as cli


def test_preserved_embedding_positional_alignment_is_validated(monkeypatch, tmp_path):
    integrated = ad.AnnData(
        X=np.ones((2, 1)),
        obs=pd.DataFrame(
            {"donor_id": ["a", "b"], "cell_type": ["x", "y"]},
            index=["input-1", "input-2"],
        ),
    )
    solution = integrated.copy()
    solution.obs_names = ["solution-1", "solution-2"]
    scored = pd.DataFrame({"method": ["example"], "score": [0.5]})
    saved = []

    monkeypatch.setattr(op_scib, "prepare_op_scib_environment", lambda: None)
    monkeypatch.setattr(op_scib.ad, "read_h5ad", lambda path: integrated)
    monkeypatch.setattr(op_scib, "load_op_solution", lambda dataset: solution)
    monkeypatch.setattr(
        op_scib,
        "compute_op_scib_metrics",
        lambda value, **kwargs: scored,
    )
    monkeypatch.setattr(
        op_scib, "save_op_scib_result", lambda result, path: saved.append((result, path))
    )

    result = op_scib.score_preserved_op_embedding(
        input_path=tmp_path / "embedding.h5ad",
        output_path=tmp_path / "score.csv",
        dataset="example",
        trust_positional_order=True,
    )

    assert result is scored
    assert integrated.obs_names.tolist() == solution.obs_names.tolist()
    assert saved == [(scored, tmp_path / "score.csv")]


def test_score_cli_delegates_to_package(monkeypatch):
    args = SimpleNamespace(
        input="embedding.h5ad",
        output="score.csv",
        dataset="example",
        embedding_key="X_emb",
        batch_key="donor_id",
        label_key="cell_type",
        method_id="method",
        trust_positional_order=False,
    )
    calls = []
    result = pd.DataFrame({"score": [0.5]})
    monkeypatch.setattr(
        cli,
        "score_preserved_op_embedding",
        lambda **kwargs: calls.append(kwargs) or result,
    )

    cli.main(args)

    assert calls[0]["input_path"] == "embedding.h5ad"
    assert calls[0]["output_path"] == "score.csv"
