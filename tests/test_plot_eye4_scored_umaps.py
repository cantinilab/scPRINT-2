from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

SCRIPT = Path(__file__).parents[1] / "scripts" / "plot_eye4_scored_umaps.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("plot_eye4_scored_umaps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_graph(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame(
        {
            "celltype": ["Beam A", "Pericyte"] * 12,
            "species": ["human", "mouse", "pig", "macaque_fascicularis"] * 6,
        },
        index=[f"cell-{index}" for index in range(24)],
    )
    data = ad.AnnData(X=np.zeros((24, 1), dtype=np.float32), obs=obs)
    data.obsm["X_scib_embedding"] = rng.normal(size=(24, 6)).astype(np.float32)
    sc.pp.neighbors(data, use_rep="X_scib_embedding", n_neighbors=6)
    data.write_h5ad(path)


def test_eye4_plotter_uses_persisted_scoring_graphs(tmp_path, monkeypatch):
    methods = [
        "PCA",
        "Random",
        "scPRINT-2 ZS",
        "scPRINT-2 FT",
        "TranscriptFormer",
    ]
    graph_paths = {}
    for seed, method in enumerate(methods):
        path = tmp_path / f"graph-{seed}.h5ad"
        _write_graph(path, seed)
        graph_paths[method] = path
    output = tmp_path / "plots"
    argv = [str(SCRIPT)]
    for method in methods:
        argv.extend(["--graph", f"{method}={graph_paths[method]}"])
    argv.extend(["--output-dir", str(output)])
    monkeypatch.setattr(sys, "argv", argv)

    _load_script().main()

    assert (output / "eye4_scored_umaps_overview.png").stat().st_size > 0
    assert (output / "eye4_scored_umaps_celltype.pdf").stat().st_size > 0
    assert (output / "eye4_scored_umaps_species.png").stat().st_size > 0
    assert (output / "UMAPS.COMPLETE").read_text() == "COMPLETE\n"
    metadata = json.loads(
        (output / "eye4_scored_umaps.metadata.json").read_text()
    )
    assert "neighbors not recomputed" in metadata["protocol"]
    assert set(metadata["methods"]) == set(methods)
