"""OpenProblems-compatible scIB metrics for embedding outputs.

This module mirrors the OpenProblems batch-integration metric calls while
remaining usable without the OpenProblems Docker images.  It intentionally
uses ``scib`` (not ``scib_metrics.Benchmarker``), because the two packages do
not implement several metrics in the same way.
"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scib
from scib import metrics as scib_metrics
from scib.metrics.lisi import lisi_graph_py


OP_RESOLUTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
OP_COLUMNS = (
    "ari",
    "asw_batch",
    "asw_label",
    "cell_cycle_conservation",
    "clisi",
    "graph_connectivity",
    "hvg_overlap",
    "ilisi",
    "isolated_label_asw",
    "isolated_label_f1",
    "kbet",
    "nmi",
    "pcr",
)


def check_op_scib_environment(*, require_kbet: bool = True) -> dict[str, str]:
    """Check the non-Docker dependencies and return their versions.

    LISI needs a C++ compiler only on platforms for which the wheel's bundled
    executable is incompatible. kBET additionally needs R, rpy2, anndata2ri,
    and the R package kBET.
    """
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scib": getattr(scib, "__version__", "unknown"),
        "scanpy": sc.__version__,
        "anndata": ad.__version__,
    }
    if require_kbet:
        try:
            import anndata2ri  # noqa: F401
            import rpy2.robjects as ro

            ro.r("suppressPackageStartupMessages(library(kBET))")
            versions["R"] = str(ro.r("R.version.string")[0])
            versions["kBET"] = str(ro.r('as.character(packageVersion("kBET"))')[0])
        except Exception as exc:
            raise RuntimeError(
                "kBET is unavailable. Install rpy2, anndata2ri==1.3.1 and the "
                "R package kBET as documented in the notebook setup cell."
            ) from exc
    return versions


def load_op_solution(
    dataset_name: str,
    root: str | os.PathLike[str] | None = None,
) -> ad.AnnData | None:
    """Load a matching OP solution file, or return None with a clear warning.

    The lookup accepts either ``<root>/<slug>_solution.h5ad`` or
    ``<root>/<dataset_name>/output_solution.h5ad``. ``root`` defaults to the
    ``OP_SOLUTION_ROOT`` environment variable.
    """
    configured_root = root or os.environ.get("OP_SOLUTION_ROOT")
    if not configured_root:
        warnings.warn(
            "OP_SOLUTION_ROOT is unset; PCR and cell-cycle conservation will be NaN.",
            stacklevel=2,
        )
        return None
    base = Path(configured_root).expanduser()
    slug = dataset_name.rstrip("/").rsplit("/", 1)[-1]
    candidates = (
        base / f"{slug}_solution.h5ad",
        base / dataset_name / "output_solution.h5ad",
        base / slug / "output_solution.h5ad",
    )
    for candidate in candidates:
        if candidate.exists():
            return ad.read_h5ad(candidate)
    warnings.warn(
        "No OpenProblems solution found for "
        f"{dataset_name!r}; tried: {', '.join(map(str, candidates))}",
        stacklevel=2,
    )
    return None


def _native_lisi_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    cache_base = Path(cache_dir or os.environ.get("SCIB_NATIVE_CACHE", "~/.cache/scib-native"))
    return cache_base.expanduser() / f"{platform.system()}-{platform.machine()}"


def ensure_native_lisi_binary(
    cache_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Compile scIB's LISI helper for the current machine and cache it.

    scIB 1.1.7 ships an ELF/x86-64 helper in some installations, which cannot
    run on macOS/arm64. Compiling the C++ source locally preserves the original
    scIB/OpenProblems algorithm without Docker.
    """
    root = _native_lisi_root(cache_dir)
    binary = root / "scib" / "knn_graph" / "knn_graph.o"
    if binary.exists() and os.access(binary, os.X_OK):
        return binary

    source = Path(scib.__file__).resolve().parent / "knn_graph" / "knn_graph.cpp"
    if not source.exists():
        raise FileNotFoundError(f"scIB LISI source not found: {source}")
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found; set CXX or install c++/g++.")

    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [compiler, "-std=c++11", "-O3", str(source), "-o", str(binary)],
        check=True,
    )
    binary.chmod(0o755)
    return binary


def prepare_op_scib_environment(
    *,
    require_kbet: bool = True,
    lisi_cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Validate R/Python dependencies and prepare the native LISI helper."""
    versions = check_op_scib_environment(require_kbet=require_kbet)
    versions["lisi_binary"] = str(ensure_native_lisi_binary(lisi_cache_dir))
    return versions


@contextlib.contextmanager
def _native_lisi_scib_root(cache_dir: str | os.PathLike[str] | None = None):
    """Temporarily point scIB's hard-coded LISI lookup at the native binary."""
    binary = ensure_native_lisi_binary(cache_dir)
    original_file = scib.__file__
    scib.__file__ = str(binary.parents[1] / "__init__.py")
    try:
        yield
    finally:
        scib.__file__ = original_file


def _align_solution(
    integrated: ad.AnnData,
    solution: ad.AnnData | None,
    *,
    batch_key: str,
    label_key: str,
) -> tuple[ad.AnnData, ad.AnnData | None]:
    result = integrated.copy()
    if solution is None:
        if batch_key not in result.obs or label_key not in result.obs:
            raise KeyError(f"integrated.obs must contain {batch_key!r} and {label_key!r}")
        return result, None

    missing = result.obs_names.difference(solution.obs_names)
    if len(missing):
        raise ValueError(f"{len(missing)} integrated cells are absent from the solution")
    aligned = solution[result.obs_names].copy()
    for key in (batch_key, label_key):
        if key not in aligned.obs:
            raise KeyError(f"solution.obs is missing {key!r}")
        result.obs[key] = aligned.obs[key].astype("category")
    result.uns.update(aligned.uns)
    return result, aligned


def _metric(
    name: str,
    fn,
    errors: dict[str, str],
    *,
    strict: bool,
) -> float:
    try:
        value = fn()
        return float(value) if value is not None else np.nan
    except Exception as exc:
        errors[name] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise
        warnings.warn(f"{name} could not be computed: {errors[name]}", stacklevel=2)
        return np.nan


def compute_op_scib_metrics(
    integrated: ad.AnnData,
    *,
    embedding_key: str,
    batch_key: str = "donor_id",
    label_key: str = "cell_type",
    solution: ad.AnnData | None = None,
    organism: str | None = None,
    method_id: str | None = None,
    n_neighbors: int = 15,
    lisi_n_neighbors: int = 90,
    lisi_cache_dir: str | os.PathLike[str] | None = None,
    compute_kbet: bool = True,
    compute_expression_metrics: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute the OpenProblems scIB metrics for one embedding.

    Parameters
    ----------
    integrated
        AnnData containing the model embedding in ``obsm[embedding_key]``.
    solution
        The matching OpenProblems solution AnnData (normalized expression,
        ``batch_hvg`` and dataset metadata). It is required for PCR and cell
        cycle conservation. If omitted, these two values remain NaN rather
        than being approximated from a different expression representation.

    Notes
    -----
    ``hvg_overlap`` is NaN by design for embedding outputs in OpenProblems.
    The returned table has the exact OpenProblems metric identifiers. Failures
    are recorded in ``DataFrame.attrs['errors']`` unless ``strict=True``.
    """
    if embedding_key not in integrated.obsm:
        raise KeyError(f"integrated.obsm is missing {embedding_key!r}")

    work, solution_aligned = _align_solution(
        integrated, solution, batch_key=batch_key, label_key=label_key
    )
    work.obsm["X_emb"] = np.asarray(work.obsm[embedding_key], dtype=np.float32)
    work.obs["batch"] = work.obs[batch_key].astype("category")
    work.obs["cell_type"] = work.obs[label_key].astype("category")
    sc.pp.neighbors(work, n_neighbors=n_neighbors, use_rep="X_emb")

    errors: dict[str, str] = {}
    values: dict[str, Any] = {key: np.nan for key in OP_COLUMNS}

    def clustering_scores() -> tuple[float, float]:
        scib_metrics.cluster_optimal_resolution(
            adata=work,
            label_key="cell_type",
            cluster_key="leiden",
            cluster_function=sc.tl.leiden,
            resolutions=list(OP_RESOLUTIONS),
            verbose=verbose,
        )
        return (
            scib_metrics.ari(work, cluster_key="leiden", label_key="cell_type"),
            scib_metrics.nmi(work, cluster_key="leiden", label_key="cell_type"),
        )

    try:
        values["ari"], values["nmi"] = clustering_scores()
    except Exception as exc:
        errors["ari/nmi"] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise
        warnings.warn(f"ari/nmi could not be computed: {errors['ari/nmi']}", stacklevel=2)

    values["asw_label"] = _metric(
        "asw_label",
        lambda: scib_metrics.silhouette(work, label_key="cell_type", embed="X_emb"),
        errors,
        strict=strict,
    )
    values["asw_batch"] = _metric(
        "asw_batch",
        lambda: scib_metrics.silhouette_batch(
            work, batch_key="batch", label_key="cell_type", embed="X_emb", verbose=verbose
        ),
        errors,
        strict=strict,
    )
    values["graph_connectivity"] = _metric(
        "graph_connectivity",
        lambda: scib_metrics.graph_connectivity(work, label_key="cell_type"),
        errors,
        strict=strict,
    )
    values["isolated_label_asw"] = _metric(
        "isolated_label_asw",
        lambda: scib_metrics.isolated_labels_asw(
            work,
            label_key="cell_type",
            batch_key="batch",
            embed="X_emb",
            iso_threshold=None,
            verbose=verbose,
        ),
        errors,
        strict=strict,
    )
    values["isolated_label_f1"] = _metric(
        "isolated_label_f1",
        lambda: scib_metrics.isolated_labels_f1(
            work,
            label_key="cell_type",
            batch_key="batch",
            cluster_key="leiden",
            resolutions=list(OP_RESOLUTIONS),
            embed=None,
            iso_threshold=None,
            verbose=verbose,
        ),
        errors,
        strict=strict,
    )

    def lisi_scores() -> tuple[float, float]:
        with _native_lisi_scib_root(lisi_cache_dir):
            ilisi_raw = lisi_graph_py(
                work, "batch", n_neighbors=lisi_n_neighbors, n_cores=1, verbose=verbose
            )
            clisi_raw = lisi_graph_py(
                work, "cell_type", n_neighbors=lisi_n_neighbors, n_cores=1, verbose=verbose
            )
        n_batches = work.obs["batch"].nunique()
        n_labels = work.obs["cell_type"].nunique()
        ilisi = (np.nanmedian(ilisi_raw) - 1) / (n_batches - 1)
        clisi = (n_labels - np.nanmedian(clisi_raw)) / (n_labels - 1)
        return float(ilisi), float(clisi)

    try:
        values["ilisi"], values["clisi"] = lisi_scores()
    except Exception as exc:
        errors["ilisi/clisi"] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise
        warnings.warn(f"ilisi/clisi could not be computed: {errors['ilisi/clisi']}", stacklevel=2)

    if compute_kbet:
        values["kbet"] = _metric(
            "kbet",
            lambda: scib_metrics.kBET(
                work,
                batch_key="batch",
                label_key="cell_type",
                type_="embed",
                embed="X_emb",
                scaled=True,
                verbose=verbose,
            ),
            errors,
            strict=strict,
        )

    if compute_expression_metrics and solution_aligned is not None:
        pre = solution_aligned.copy()
        pre.obs["batch"] = pre.obs[batch_key].astype("category")
        post = work.copy()
        post.obs["batch"] = post.obs[batch_key].astype("category")
        hvg_mask = pre.var["batch_hvg"].astype(bool) if "batch_hvg" in pre.var else slice(None)
        values["pcr"] = _metric(
            "pcr",
            lambda: scib_metrics.pcr_comparison(
                pre[:, hvg_mask], post, embed="X_emb", covariate="batch", verbose=verbose
            ),
            errors,
            strict=strict,
        )
        organism_value = organism or pre.uns.get("dataset_organism")
        if organism_value is None:
            errors["cell_cycle_conservation"] = "dataset organism is unavailable"
        else:
            if "feature_name" in pre.var:
                pre.var_names = pre.var["feature_name"].astype(str)
            values["cell_cycle_conservation"] = _metric(
                "cell_cycle_conservation",
                lambda: scib_metrics.cell_cycle(
                    pre,
                    post,
                    batch_key="batch",
                    embed="X_emb",
                    organism=str(organism_value),
                    verbose=verbose,
                ),
                errors,
                strict=strict,
            )
    elif compute_expression_metrics:
        errors["pcr"] = "OpenProblems solution AnnData not provided"
        errors["cell_cycle_conservation"] = "OpenProblems solution AnnData not provided"

    index = method_id or embedding_key
    result = pd.DataFrame([values], index=pd.Index([index], name="method"))[list(OP_COLUMNS)]
    result.attrs["errors"] = errors
    result.attrs["parameters"] = {
        "neighbors": n_neighbors,
        "lisi_neighbors": lisi_n_neighbors,
        "resolutions": list(OP_RESOLUTIONS),
        "embedding_key": embedding_key,
        "batch_key": batch_key,
        "label_key": label_key,
    }
    return result


def save_op_scib_result(result: pd.DataFrame, path: str | os.PathLike[str]) -> Path:
    """Save a metric row and a sidecar containing parameters and failures."""
    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    sidecar = output.with_suffix(output.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(
            {
                "parameters": result.attrs.get("parameters", {}),
                "errors": result.attrs.get("errors", {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output
