"""Reusable Task3 scIB preparation and scoring helpers."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

PAPER_SCIB_VERSION = "1.1.3"
PAPER_SCANPY_VERSION = "1.9.1"
TASK3_SCIB113_EXPECTED_VERSIONS = {
    "scib": PAPER_SCIB_VERSION,
    "scanpy": PAPER_SCANPY_VERSION,
}
BATCH_METRICS = ["ASW_label/batch", "PCR_batch", "graph_conn", "kBET", "iLISI"]
BIO_METRICS = [
    "NMI_cluster/label",
    "ARI_cluster/label",
    "ASW_label",
    "cLISI",
    "hvg_overlap",
    "trajectory",
]


@dataclass(frozen=True)
class Task3Scib113Config:
    """Python-facing configuration for the paper-compatible Task3 scorer."""

    source: Path | str
    embedding: tuple[tuple[str, Path | str, str], ...]
    output: Path | str
    batch_key: str = "orig.ident"
    label_key: str = "cell_type_ontology_term_id"
    n_cores: int = 8
    obs_name_replace: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if not self.embedding:
            raise ValueError("At least one Task3 embedding is required")
        names = [name for name, _, _ in self.embedding]
        if len(names) != len(set(names)):
            raise ValueError("Task3 embedding method names must be unique")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["source"] = str(values["source"])
        values["output"] = str(values["output"])
        values["embedding"] = [
            {"method": method, "path": str(path), "key": key}
            for method, path, key in self.embedding
        ]
        return values

    def as_namespace(self):
        from argparse import Namespace

        return Namespace(
            source=Path(self.source),
            embedding=[
                (method, Path(path), key) for method, path, key in self.embedding
            ],
            output=Path(self.output),
            batch_key=self.batch_key,
            label_key=self.label_key,
            n_cores=self.n_cores,
            obs_name_replace=self.obs_name_replace,
        )


@dataclass
class Task3Scib113Result:
    table: pd.DataFrame
    output: Path
    metadata: Path
    complete: Path

    def artifact_table(self) -> pd.Series:
        return pd.Series(
            {
                "scores": str(self.output),
                "metadata": str(self.metadata),
                "complete": str(self.complete),
            },
            name="artifacts",
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sparse_sha256(matrix: sp.spmatrix) -> str:
    csr = matrix.tocsr()
    digest = hashlib.sha256()
    digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
    digest.update(csr.indptr.tobytes())
    digest.update(csr.indices.tobytes())
    digest.update(csr.data.tobytes())
    return digest.hexdigest()


def task3_scib_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in (
            "scib",
            "scanpy",
            "anndata",
            "numpy",
            "pandas",
            "scikit-learn",
        )
    }


def assert_task3_scib113_environment(
    *, mismatch_message: str = "Unexpected scoring environment"
) -> dict[str, str]:
    versions = task3_scib_versions()
    mismatches = {
        package: (versions[package], expected)
        for package, expected in TASK3_SCIB113_EXPECTED_VERSIONS.items()
        if versions[package] != expected
    }
    if mismatches:
        raise RuntimeError(f"{mismatch_message}: {mismatches}")
    return versions


def parse_embedding_spec(specification: str) -> tuple[str, Path, str]:
    method, separator, remainder = specification.partition("=")
    path, key_separator, key = remainder.rpartition(":")
    if not separator or not key_separator or not method or not path or not key:
        raise ValueError(
            f"Invalid embedding {specification!r}; expected METHOD=PATH:OBSM_KEY"
        )
    return method, Path(path), key


def parse_obs_name_replace_spec(specification: str) -> tuple[str, str]:
    source_suffix, separator, embedding_suffix = specification.partition("=")
    if not separator or not source_suffix or not embedding_suffix:
        raise ValueError("obs-name-replace must be SOURCE_SUFFIX=EMBEDDING_SUFFIX")
    return source_suffix, embedding_suffix


def aligned_embedding(
    source: ad.AnnData,
    result: ad.AnnData,
    key: str,
    *,
    missing_message: str = "embedding output",
    unaligned_message: str = "Embedding output cannot be aligned to task3 source cell IDs",
    invalid_message: str = "Embedding has the wrong number of rows or non-finite values",
) -> np.ndarray:
    if key not in result.obsm:
        raise KeyError(f"Missing obsm[{key!r}] in {missing_message}")
    if source.obs_names.equals(result.obs_names):
        aligned = result
    elif source.obs_names.isin(result.obs_names).all():
        aligned = result[source.obs_names]
    else:
        raise RuntimeError(unaligned_message)
    embedding = np.asarray(aligned.obsm[key], dtype=np.float32)
    if embedding.shape[0] != source.n_obs or not np.isfinite(embedding).all():
        if "{shape}" in invalid_message:
            invalid_message = invalid_message.format(shape=embedding.shape)
        raise RuntimeError(invalid_message)
    return embedding


def load_aligned_embedding(
    source: ad.AnnData,
    path: Path,
    key: str,
    obs_name_replace: tuple[str, str] | None = None,
) -> np.ndarray:
    result = ad.read_h5ad(path, backed="r")
    if key not in result.obsm:
        raise KeyError("Missing obsm[{!r}] in {}".format(key, path))
    if source.obs_names.equals(result.obs_names):
        order = source.obs_names
    elif source.obs_names.isin(result.obs_names).all():
        order = source.obs_names
    elif obs_name_replace is not None:
        source_suffix, embedding_suffix = obs_name_replace
        if not source_suffix or not source.obs_names.str.endswith(source_suffix).all():
            raise RuntimeError(
                "Source obs names do not all end with {!r}".format(source_suffix)
            )
        order = pd.Index(
            [
                name[: -len(source_suffix)] + embedding_suffix
                for name in source.obs_names
            ]
        )
        if not order.is_unique or not order.isin(result.obs_names).all():
            raise RuntimeError(
                "Mapped obs names do not align {} to source cells".format(path)
            )
    else:
        raise RuntimeError("Cannot align {} to task3 cells".format(path))
    values = np.asarray(result[order].obsm[key], dtype=np.float32)
    if values.shape[0] != source.n_obs or not np.isfinite(values).all():
        raise RuntimeError("Invalid embedding {}:{} {}".format(path, key, values.shape))
    return values


def clear_dimensionality_reduction(adata: ad.AnnData) -> None:
    for key in ("connectivities", "distances"):
        if key in adata.obsp:
            del adata.obsp[key]
    for key in ("neighbors", "pca"):
        if key in adata.uns:
            del adata.uns[key]
    if "X_pca" in adata.obsm:
        del adata.obsm["X_pca"]
    if "PCs" in adata.varm:
        del adata.varm["PCs"]


def precompute_reference_pseudotime(
    prepared: ad.AnnData, batch_key: str
) -> dict[str, object]:
    pseudotime = pd.Series(np.nan, index=prepared.obs_names, dtype=float)
    batches: dict[str, object] = {}
    for batch in prepared.obs[batch_key].cat.categories:
        mask = prepared.obs[batch_key] == batch
        per_batch = prepared[mask].copy()
        if per_batch.n_obs < 3:
            raise RuntimeError(
                "Cannot compute reference pseudotime for batch {!r}: {} cells".format(
                    batch, per_batch.n_obs
                )
            )
        sc.pp.highly_variable_genes(per_batch)
        n_highly_variable = int(per_batch.var["highly_variable"].sum())
        sc.tl.pca(per_batch, use_highly_variable=n_highly_variable >= 2)
        sc.pp.neighbors(per_batch, use_rep="X_pca")
        per_batch.uns["iroot"] = 0
        sc.tl.diffmap(per_batch)
        sc.tl.dpt(per_batch)
        values = per_batch.obs["dpt_pseudotime"].astype(float)
        finite = np.isfinite(values.to_numpy())
        if not finite.any():
            raise RuntimeError(
                "Reference pseudotime is unavailable for every cell in batch "
                "{!r}".format(batch)
            )
        pseudotime.loc[per_batch.obs_names] = values.to_numpy()
        batches[str(batch)] = {
            "cells": int(per_batch.n_obs),
            "genes": int(per_batch.n_vars),
            "highly_variable_genes": n_highly_variable,
            "root_position": 0,
            "finite_pseudotime_cells": int(finite.sum()),
            "missing_pseudotime_cells": int((~finite).sum()),
        }
    # scIB 1.1.3 explicitly accepts missing reference pseudotime and subsets the
    # trajectory metric to non-null cells. Disconnected expression components
    # therefore remain unavailable instead of receiving invented distances.
    prepared.obs["dpt_pseudotime"] = pseudotime.loc[prepared.obs_names].to_numpy()
    return batches


def prepare_expression(
    source: ad.AnnData, batch_key: str
) -> tuple[ad.AnnData, dict[str, object]]:
    prepared = source.copy()
    clear_dimensionality_reduction(prepared)
    prepared.uns.pop("log1p", None)
    sc.pp.normalize_total(prepared, target_sum=10_000)
    sc.pp.log1p(prepared)
    trajectory_batches = precompute_reference_pseudotime(prepared, batch_key)
    sc.pp.pca(prepared)
    sc.pp.neighbors(prepared, use_rep="X_pca")
    return prepared, trajectory_batches


def safe_name(method: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", method).strip("_")


def write_precomputed_graph(
    output_dir: Path,
    method: str,
    integrated: ad.AnnData,
    representation_key: str,
) -> Path:
    graph = ad.AnnData(
        X=sp.csr_matrix((integrated.n_obs, 0), dtype=np.float32),
        obs=integrated.obs.copy(),
    )
    graph.obsm[representation_key] = np.asarray(
        integrated.obsm[representation_key], dtype=np.float32
    )
    graph.obsp["connectivities"] = integrated.obsp["connectivities"].copy()
    graph.obsp["distances"] = integrated.obsp["distances"].copy()
    graph.uns["neighbors"] = integrated.uns["neighbors"].copy()
    path = output_dir / "precomputed_graphs" / ("{}.h5ad".format(safe_name(method)))
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.write_h5ad(path, compression="lzf")
    return path


def mean_available(metrics: dict[str, float], names: list[str]) -> tuple[float, int]:
    values = np.asarray([metrics[name] for name in names], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return float("nan"), 0
    return float(values[finite].mean()), int(finite.sum())


def run_modern_scib_benchmark(args: Any) -> None:
    from importlib import metadata

    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    source = sc.read_h5ad(args.source)
    for column in (args.batch_key, args.label_key):
        if column not in source.obs:
            raise KeyError(f"Missing source obs column {column!r}")

    method_names = []
    provenance = {}
    for specification in args.embedding:
        method, path, key = parse_embedding_spec(specification)
        result = sc.read_h5ad(path, backed="r")
        values = aligned_embedding(
            source,
            result,
            key,
            missing_message=f"{method}",
            unaligned_message=f"Cannot align {method} to the task3 source cells",
            invalid_message=f"Invalid embedding matrix for {method}: {{shape}}",
        )
        if values.shape[0] != source.n_obs or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid embedding matrix for {method}: {values.shape}")
        source.obsm[method] = values
        method_names.append(method)
        provenance[method] = {"path": str(path.resolve()), "key": key}

    sc.pp.normalize_total(source, target_sum=10_000)
    sc.pp.log1p(source)
    sc.tl.pca(
        source,
        n_comps=50,
        svd_solver="arpack",
        use_highly_variable=False,
    )
    rng = np.random.default_rng(args.seed)
    source.obsm["random"] = rng.random(source.obsm["X_pca"].shape, dtype=np.float32)

    benchmark = Benchmarker(
        source,
        batch_key=args.batch_key,
        label_key=args.label_key,
        embedding_obsm_keys=[*method_names, "X_pca", "random"],
        pre_integrated_embedding_obsm_key="X_pca",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=10,
    )
    benchmark.benchmark()
    results = benchmark.get_results(min_max_scale=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output)
    run_metadata = {
        "protocol": "cross-species notebook modern scib-metrics benchmark",
        "versions": {
            package: metadata.version(package)
            for package in ("scib-metrics", "scanpy", "anndata", "numpy", "pandas")
        },
        "source": str(args.source.resolve()),
        "batch_key": args.batch_key,
        "label_key": args.label_key,
        "seed": args.seed,
        "embeddings": provenance,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n"
    )


def run_scib113_precomputed_benchmark(
    config: Task3Scib113Config | Any,
) -> Task3Scib113Result:
    args = config.as_namespace() if isinstance(config, Task3Scib113Config) else config
    if args.output.exists() or args.output.with_suffix(".COMPLETE").exists():
        raise FileExistsError(
            "Refusing to overwrite an existing score artifact: {}".format(args.output)
        )

    versions = assert_task3_scib113_environment()

    from scib.metrics import metrics as scib_metrics

    source = ad.read_h5ad(args.source)
    for column in (args.batch_key, args.label_key):
        if column not in source.obs:
            raise KeyError("Missing source obs[{!r}]".format(column))
        source.obs[column] = source.obs[column].astype("category")
    expression, trajectory_batches = prepare_expression(source, args.batch_key)
    expression_graph_hash = sparse_sha256(expression.obsp["connectivities"])

    rows: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for method, path, key in args.embedding:
        values = load_aligned_embedding(
            source, path, key, obs_name_replace=args.obs_name_replace
        )
        integrated = expression.copy()
        for graph_key in ("connectivities", "distances"):
            if graph_key in integrated.obsp:
                del integrated.obsp[graph_key]
        integrated.uns.pop("neighbors", None)
        representation_key = "X_scib_embedding"
        integrated.obsm[representation_key] = values
        sc.pp.neighbors(integrated, use_rep=representation_key)
        graph_hash_before = sparse_sha256(integrated.obsp["connectivities"])
        graph_path = write_precomputed_graph(
            args.output.parent, method, integrated, representation_key
        )

        result = scib_metrics(
            expression,
            integrated,
            batch_key=args.batch_key,
            label_key=args.label_key,
            embed=representation_key,
            ari_=True,
            nmi_=True,
            silhouette_=True,
            pcr_=True,
            hvg_score_=True,
            isolated_labels_f1_=True,
            isolated_labels_asw_=True,
            graph_conn_=True,
            trajectory_=True,
            kBET_=True,
            ilisi_=True,
            clisi_=True,
            type_="knn",
            subsample=0.5,
            n_cores=args.n_cores,
            verbose=True,
        )
        metric_values = {
            str(name): float(value) for name, value in result.iloc[:, 0].items()
        }
        for required_metric in ("hvg_overlap", "trajectory"):
            if not np.isfinite(metric_values[required_metric]):
                raise RuntimeError(
                    "{} is not finite for {}".format(required_metric, method)
                )
        batch_score, batch_count = mean_available(metric_values, BATCH_METRICS)
        bio_score, bio_count = mean_available(metric_values, BIO_METRICS)
        total = 0.4 * batch_score + 0.6 * bio_score
        graph_hash_after = sparse_sha256(integrated.obsp["connectivities"])
        if graph_hash_after != graph_hash_before:
            raise RuntimeError(
                "scIB modified the precomputed graph for {}".format(method)
            )
        row: dict[str, object] = {
            "Method": method,
            "Batch correction": batch_score,
            "Bio conservation": bio_score,
            "Total": total,
            "Batch metrics available": batch_count,
            "Bio metrics available": bio_count,
        }
        row.update(metric_values)
        rows.append(row)
        provenance[method] = {
            "embedding": str(path.resolve()),
            "embedding_sha256": sha256(path),
            "embedding_key": key,
            "embedding_dimension": int(values.shape[1]),
            "precomputed_graph": str(graph_path.resolve()),
            "precomputed_graph_sha256": sha256(graph_path),
            "connectivities_sha256_before_scib": graph_hash_before,
            "connectivities_sha256_after_scib": graph_hash_after,
            "neighbors_params": integrated.uns["neighbors"].get("params", {}),
        }
        del integrated, values
        gc.collect()

    table = pd.DataFrame(rows).sort_values("Total", ascending=False)
    table.to_csv(args.output, index=False)
    metadata = {
        "protocol": "scib 1.1.3 with caller-precomputed PCA and kNN graphs",
        "versions": versions,
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "batch_key": args.batch_key,
        "label_key": args.label_key,
        "obs_name_replace": args.obs_name_replace,
        "cells": int(source.n_obs),
        "labels": int(source.obs[args.label_key].nunique()),
        "batches": int(source.obs[args.batch_key].nunique()),
        "expression_preprocessing": [
            "normalize_total(target_sum=1e4)",
            "log1p",
            "per-batch highly_variable_genes",
            "per-batch pca",
            "per-batch neighbors(use_rep='X_pca')",
            "per-batch diffmap and dpt(root_position=0)",
            "pca",
            "neighbors(use_rep='X_pca')",
        ],
        "trajectory_reference": {
            "pseudotime_key": "dpt_pseudotime",
            "batches": trajectory_batches,
        },
        "expression_connectivities_sha256": expression_graph_hash,
        "integration_graph": {
            "precomputed_before_scib": True,
            "neighbors_use_rep": "X_scib_embedding",
            "scib_type": "knn",
        },
        "batch_metrics": BATCH_METRICS,
        "bio_metrics": BIO_METRICS,
        "total_formula": "0.4 * Batch correction + 0.6 * Bio conservation",
        "embeddings": provenance,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    complete_path = args.output.with_suffix(".COMPLETE")
    complete_path.write_text("COMPLETE\n")
    print(table.to_string(index=False))
    return Task3Scib113Result(
        table=table,
        output=args.output,
        metadata=metadata_path,
        complete=complete_path,
    )


def prepare_paper_scib(args: Any) -> None:
    versions = assert_task3_scib113_environment(
        mismatch_message="Paper environment mismatch"
    )
    source_path = Path(args.source).resolve()
    result_path = Path(args.embedding).resolve()
    root = Path(args.paper_root).resolve()
    method_dir = root / "output" / "method_outputs" / "saturn"
    evaluation_dir = root / "output" / "evaluation" / "saturn"
    raw_dir = root / "data" / "process_data" / "h5ad_files"
    for directory in (method_dir, evaluation_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source = sc.read_h5ad(source_path)
    result = sc.read_h5ad(result_path)
    for column in (args.batch_key, args.label_key):
        if column not in source.obs:
            raise KeyError(f"Missing source obs column {column!r}")
    embedding = aligned_embedding(source, result, args.embedding_key)

    batch = source.obs[args.batch_key].astype(str)
    labels = source.obs[args.label_key].astype(str)
    categories = sorted(batch.unique())
    if len(categories) != 2:
        raise RuntimeError(
            f"Paper task3 requires two species batches, found {categories}"
        )
    order = np.concatenate(
        [np.flatnonzero(batch.to_numpy() == value) for value in categories]
    )
    ordered_source = source[order].copy()
    ordered_embedding = embedding[order]
    ordered_batch = batch.iloc[order].to_numpy()
    ordered_labels = labels.iloc[order].to_numpy()

    for index, value in enumerate(categories):
        raw = ordered_source[ordered_batch == value].copy()
        sample = raw.X[: min(128, raw.n_obs)]
        values = sample.data if hasattr(sample, "data") else np.asarray(sample).ravel()
        if values.size and np.all(values >= 0) and np.allclose(values, np.rint(values)):
            sc.pp.normalize_total(raw, target_sum=10_000)
            sc.pp.log1p(raw)
        raw.uns.pop("log1p", None)
        raw.write_h5ad(raw_dir / f"task3_{index}_{value}.h5ad", compression="lzf")

    obs = pd.DataFrame(
        {
            "species": ordered_batch,
            "labels2": ordered_labels,
            "batch": ordered_batch,
            "celltype": ordered_labels,
        },
        index=ordered_source.obs_names.copy(),
    )
    integrated = ad.AnnData(X=ordered_embedding, obs=obs)
    integrated.write_h5ad(method_dir / "task3.h5ad", compression="lzf")

    table = pd.DataFrame(
        ordered_embedding,
        index=ordered_source.obs_names,
        columns=[f"dim_{i}" for i in range(ordered_embedding.shape[1])],
    )
    table["celltype"] = ordered_labels
    table["batchlb"] = ordered_batch
    table.to_csv(method_dir / "task3_saturn_embedding.txt")

    metadata = {
        "protocol": "NAR gkae1316 Figshare file 50760384",
        "paper_versions": versions,
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "embedding": str(result_path),
        "embedding_sha256": sha256(result_path),
        "embedding_key": args.embedding_key,
        "batch_key": args.batch_key,
        "label_key": args.label_key,
        "batch_order": categories,
        "n_obs": int(source.n_obs),
        "embedding_dim": int(ordered_embedding.shape[1]),
    }
    (root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
