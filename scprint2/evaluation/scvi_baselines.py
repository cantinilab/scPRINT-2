"""Reusable scVI baseline runners for OpenProblems and task3 controls."""

from __future__ import annotations

import importlib.metadata as package_metadata
import json
import random
from argparse import Namespace
from pathlib import Path
from typing import Any

OPENPROBLEMS_SCVI_VERSION_PACKAGES = ("anndata", "scvi-tools", "torch")

TASK3_PAPER_SCVI_EXPECTED_VERSIONS = {
    "scvi-tools": "0.19.0",
    "scanpy": "1.9.1",
    "anndata": "0.8.0",
    "numpy": "1.21.6",
    "pandas": "1.3.5",
    "scikit-learn": "1.0.2",
    "torch": "1.12.1+cu102",
}


def package_versions(packages: tuple[str, ...]) -> dict[str, str]:
    """Return installed package versions used in metadata provenance."""
    return {package: package_metadata.version(package) for package in packages}


def task3_paper_scvi_versions() -> dict[str, str]:
    """Validate the archived scVI paper environment before running task3."""
    versions = package_versions(tuple(TASK3_PAPER_SCVI_EXPECTED_VERSIONS))
    mismatches = {
        package: (versions[package], expected)
        for package, expected in TASK3_PAPER_SCVI_EXPECTED_VERSIONS.items()
        if versions[package] != expected
    }
    if mismatches:
        raise RuntimeError(f"Paper scVI environment mismatch: {mismatches}")
    return versions


def write_json_metadata(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write deterministic JSON metadata and return its path."""
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return metadata_path


def run_openproblems_scvi(args: Namespace) -> None:
    """Run the OpenProblems scVI recipe and preserve its latent representation."""
    import anndata as ad
    import scvi
    from scvi.model import SCVI

    if args.seed is not None:
        scvi.settings.seed = args.seed
    adata = ad.read_h5ad(args.input)
    adata.X = adata.layers["counts"].copy()
    idx = adata.var["hvg_score"].to_numpy().argsort()[::-1][: args.n_hvg]
    adata = adata[:, idx].copy()

    SCVI.setup_anndata(adata, batch_key="batch")
    vae = SCVI(
        adata,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        n_layers=args.n_layers,
    )
    vae.train(max_epochs=args.max_epochs, train_size=1.0)

    output = ad.AnnData(
        obs=adata.obs.copy(),
        obsm={"X_emb": vae.get_latent_representation()},
        uns={
            "openproblems_scvi_parameters": {
                "n_hvg": args.n_hvg,
                "n_latent": args.n_latent,
                "n_hidden": args.n_hidden,
                "n_layers": args.n_layers,
                "max_epochs": args.max_epochs,
                "train_size": 1.0,
                "batch_key": "batch",
                "seed": args.seed,
            },
            "software_versions": package_versions(OPENPROBLEMS_SCVI_VERSION_PACKAGES),
        },
    )
    output.write_h5ad(args.output, compression="gzip")
    write_json_metadata(f"{args.output}.meta.json", output.uns)


def run_task3_paper_scvi(args: Namespace) -> None:
    """Run the task3 scVI control with the archived paper package versions."""
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scvi
    import torch

    versions = task3_paper_scvi_versions()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source = sc.read_h5ad(args.input)
    for column in (args.batch_key, args.label_key):
        if column not in source.obs:
            raise KeyError(f"Missing source obs column {column!r}")
    batches = sorted(source.obs[args.batch_key].astype(str).unique())
    if len(batches) != 2:
        raise RuntimeError(f"Paper task3 requires two batches, found {batches}")

    split = []
    for batch in batches:
        current = source[source.obs[args.batch_key].astype(str) == batch].copy()
        current.obs["batch"] = current.obs[args.batch_key].astype(str)
        current.obs["celltype"] = current.obs[args.label_key].astype(str)
        sc.pp.filter_cells(current, min_genes=200)
        sc.pp.filter_genes(current, min_cells=10)
        current.layers["counts"] = current.X.copy()
        split.append(current)

    integrated = ad.concat(split, join="inner")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    scvi.settings.seed = args.seed

    sc.pp.normalize_total(integrated, target_sum=10_000)
    sc.pp.log1p(integrated)
    integrated.raw = integrated
    integrated.obs["batch"] = integrated.obs["batch"].astype(str)
    sc.pp.highly_variable_genes(
        integrated,
        n_top_genes=2000,
        subset=True,
        layer="counts",
        flavor="seurat_v3",
        batch_key="batch",
    )
    scvi.model.SCVI.setup_anndata(
        integrated,
        layer="counts",
        batch_key="batch",
    )
    model = scvi.model.SCVI(integrated, n_latent=30)
    # The archived torch 1.12.1+cu102 wheel only contains kernels through
    # compute capability 7.0. Jean Zay exposes newer GPUs, so CPU preserves the
    # exact paper stack without CUDA kernel mismatch.
    model.train(use_gpu=False)
    integrated.obsm["X_scVI"] = model.get_latent_representation()
    integrated.obs["orig.ident"] = integrated.obs["batch"]
    integrated.write_h5ad(output, compression="lzf")
    model.save(output.with_suffix(".model"), overwrite=False)
    write_json_metadata(
        output.with_suffix(".metadata.json"),
        {
            "protocol": "NAR gkae1316 Figshare file 50760384 scVI_for_all_tasks.py",
            "versions": versions,
            "seed": args.seed,
            "input": str(Path(args.input).resolve()),
            "output": str(output),
            "batch_order": batches,
            "n_obs": int(integrated.n_obs),
            "n_vars": int(integrated.n_vars),
            "embedding_dim": int(integrated.obsm["X_scVI"].shape[1]),
            "accelerator": "cpu",
        },
    )
