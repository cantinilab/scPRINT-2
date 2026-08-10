"""OpenProblems-compatible scIB metrics for embedding outputs.

This module mirrors the OpenProblems batch-integration metric calls while
remaining usable without the OpenProblems Docker images.  It intentionally
uses ``scib`` (not ``scib_metrics.Benchmarker``), because the two packages do
not implement several metrics in the same way.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
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
from scipy import sparse

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

OP_LOG_CP10K_TARGET_SUM = 10_000.0
OP_BATCH_HVGS = 2_000

# Published OpenProblems batch-integration values, rounded to four decimals.
# These are used only as a validation target; they are never used to construct
# the reconstructed reference or any metric input.
OP_NO_INTEGRATION_EXPECTED = {
    "dkd": {
        "ari": 0.5999,
        "asw_batch": 0.8913,
        "asw_label": 0.6276,
        "cell_cycle_conservation": 0.8248,
        "clisi": 0.9998,
        "graph_connectivity": 0.9701,
        "hvg_overlap": np.nan,
        "ilisi": 0.0754,
        "isolated_label_asw": np.nan,
        "isolated_label_f1": np.nan,
        "kbet": 0.1529,
        "nmi": 0.7735,
        "pcr": 0.0,
    }
}


def _software_versions() -> dict[str, str]:
    packages = (
        "anndata",
        "igraph",
        "leidenalg",
        "numpy",
        "pandas",
        "pynndescent",
        "scanpy",
        "scib",
        "scib-metrics",
        "scikit-learn",
        "scipy",
        "umap-learn",
    )
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _matrix_max_abs(matrix: Any) -> float:
    """Return the largest absolute entry without densifying sparse matrices."""
    if sparse.issparse(matrix):
        return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
    values = np.asarray(matrix)
    return float(np.max(np.abs(values))) if values.size else 0.0


def normalize_log_cp10k_from_counts(
    adata: ad.AnnData,
    *,
    counts_layer: str = "counts",
    target_sum: float = OP_LOG_CP10K_TARGET_SUM,
) -> tuple[Any, np.ndarray]:
    """Reproduce the OpenProblems ``log_cp10k`` normalization from counts.

    This deliberately ignores any existing normalized layer. It matches the
    OpenProblems common-dataset processor: ``scanpy.pp.normalize_total`` on
    ``layers['counts']`` followed by ``scanpy.pp.log1p``.
    """
    if counts_layer not in adata.layers:
        raise KeyError(f"adata.layers is missing {counts_layer!r}")
    normalized = sc.pp.normalize_total(
        adata,
        target_sum=target_sum,
        layer=counts_layer,
        inplace=False,
    )
    log_normalized = sc.pp.log1p(normalized["X"])
    return log_normalized, np.asarray(normalized["norm_factor"])


def reconstruct_op_batch_reference(
    common: ad.AnnData,
    *,
    n_hvgs: int = OP_BATCH_HVGS,
    target_sum: float = OP_LOG_CP10K_TARGET_SUM,
    verify_published_normalized: bool = True,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Rebuild an OpenProblems batch-integration reference from raw counts.

    The input is the public OpenProblems *common dataset*, not the task
    solution. The function reruns log-CP10k normalization, batch-aware HVG
    selection, PCA and the 30-neighbour dataset graph using the v2.0.0 task
    processor's calls. The returned AnnData has the fields needed by PCR and
    cell-cycle conservation.
    """
    required_obs = {"batch", "cell_type"}
    missing_obs = required_obs.difference(common.obs)
    if missing_obs:
        raise KeyError(f"common.obs is missing: {sorted(missing_obs)}")
    if "counts" not in common.layers:
        raise KeyError("common.layers is missing 'counts'")

    reference = common.copy()
    published_normalized = reference.layers.get("normalized")
    normalized, size_factors = normalize_log_cp10k_from_counts(
        reference, target_sum=target_sum
    )

    normalization_check: dict[str, Any] = {
        "target_sum": float(target_sum),
        "published_layer_present": published_normalized is not None,
    }
    if published_normalized is not None and verify_published_normalized:
        delta = normalized - published_normalized
        normalization_check.update(
            {
                "published_max_abs": _matrix_max_abs(published_normalized),
                "recomputed_max_abs": _matrix_max_abs(normalized),
                "max_abs_difference": _matrix_max_abs(delta),
            }
        )

    reference.layers["normalized"] = normalized
    reference.obs["size_factors"] = size_factors
    reference.uns["normalization_id"] = "log_cp10k"

    n_hvgs = min(int(n_hvgs), reference.n_vars)
    if n_hvgs == reference.n_vars:
        hvg_list = reference.var_names.tolist()
    else:
        scib_adata = reference.copy()
        del scib_adata.layers["counts"]
        scib_adata.X = scib_adata.layers["normalized"].copy()
        hvg_list = scib.pp.hvg_batch(
            scib_adata,
            batch_key="batch",
            target_genes=n_hvgs,
            adataOut=False,
        )
        del scib_adata
    reference.var["batch_hvg"] = reference.var_names.isin(hvg_list)

    n_components = (
        int(common.obsm["X_pca"].shape[1]) if "X_pca" in common.obsm else 50
    )
    x_pca, loadings, variance, variance_ratio = sc.pp.pca(
        reference.layers["normalized"],
        n_comps=n_components,
        mask_var=reference.var["batch_hvg"],
        return_info=True,
    )
    reference.obsm["X_pca"] = x_pca
    reference.varm["pca_loadings"] = np.zeros(
        (reference.n_vars, n_components), dtype=np.asarray(loadings).dtype
    )
    reference.varm["pca_loadings"][reference.var["batch_hvg"], :] = loadings.T
    reference.uns["pca_variance"] = {
        "variance": variance,
        "variance_ratio": variance_ratio,
    }

    reference.uns.pop("knn", None)
    reference.obsp.pop("knn_connectivities", None)
    reference.obsp.pop("knn_distances", None)
    sc.pp.neighbors(reference, use_rep="X_pca", n_neighbors=30, key_added="knn")

    hvg = sc.pp.highly_variable_genes(
        reference,
        layer="normalized",
        n_top_genes=n_hvgs,
        flavor="cell_ranger",
        inplace=False,
    )
    reference.var["hvg"] = hvg["highly_variable"].values
    reference.var["hvg_score"] = hvg["dispersions_norm"].values

    normalization_check.update(
        {
            "n_obs": reference.n_obs,
            "n_vars": reference.n_vars,
            "n_batch_hvgs": int(reference.var["batch_hvg"].sum()),
            "n_components": n_components,
        }
    )
    return reference, normalization_check


def make_op_no_integration(reference: ad.AnnData) -> ad.AnnData:
    """Create the exact OpenProblems no-integration embedding (task PCA)."""
    if "X_pca" not in reference.obsm:
        raise KeyError("reference.obsm is missing 'X_pca'")
    output = ad.AnnData(obs=pd.DataFrame(index=reference.obs_names.copy()))
    output.obsm["X_emb"] = np.asarray(reference.obsm["X_pca"])
    output.uns.update(
        {
            key: reference.uns[key]
            for key in ("dataset_id", "normalization_id")
            if key in reference.uns
        }
    )
    output.uns["method_id"] = "no_integration"
    return output


def compare_op_scores(
    observed: pd.DataFrame,
    expected: dict[str, float],
) -> pd.DataFrame:
    """Return a metric-by-metric comparison against rounded OP scores."""
    row = observed.iloc[0]
    comparison = pd.DataFrame(
        {
            "expected": pd.Series(expected, dtype=float),
            "observed": row.reindex(expected).astype(float),
        }
    )
    comparison["absolute_difference"] = (
        comparison["observed"] - comparison["expected"]
    ).abs()
    comparison["matches_published_4dp"] = (
        comparison["observed"].round(4) == comparison["expected"].round(4)
    ) | (comparison["observed"].isna() & comparison["expected"].isna())
    return comparison


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
        if result.n_obs != solution.n_obs:
            raise ValueError(
                f"{len(missing)} integrated cells are absent from the solution "
                f"and observation counts differ ({result.n_obs} != {solution.n_obs})"
            )

        identity_columns = [
            batch_key,
            label_key,
            "cell_type_ontology_term_id",
            "assay_ontology_term_id",
            "disease_ontology_term_id",
            "sex_ontology_term_id",
            "tissue_ontology_term_id",
        ]
        compared = []
        for key in dict.fromkeys(identity_columns):
            if key not in result.obs or key not in solution.obs:
                continue
            compared.append(key)
            left = result.obs[key].astype("string").fillna("<NA>").to_numpy()
            right = solution.obs[key].astype("string").fillna("<NA>").to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(
                    "Integrated and solution observation names differ, and "
                    f"positional identity check failed for obs[{key!r}]"
                )
        if batch_key not in compared or label_key not in compared:
            raise ValueError(
                "Integrated and solution observation names differ without shared "
                "batch and label columns for positional validation"
            )

        if "size_factors" not in solution.obs:
            raise ValueError(
                "Integrated and solution observation names differ, and the solution "
                "has no size_factors fingerprint for positional validation"
            )
        reference_totals = (
            solution.obs["size_factors"].to_numpy(dtype=float)
            * OP_LOG_CP10K_TARGET_SUM
        )
        correlations = {}
        for key in ("nCount_RNA", "total_counts"):
            if key not in result.obs:
                continue
            candidate = result.obs[key].to_numpy(dtype=float)
            correlation = float(np.corrcoef(candidate, reference_totals)[0, 1])
            if np.isfinite(correlation):
                correlations[key] = correlation
        if not correlations or max(correlations.values()) < 0.999:
            raise ValueError(
                "Integrated and solution observation names differ, and library-size "
                f"fingerprints do not prove positional identity: {correlations}"
            )

        aligned = solution.copy()
        aligned.obs_names = result.obs_names.copy()
        warnings.warn(
            "Aligned solution by position after exact metadata and library-size "
            f"validation (best correlation={max(correlations.values()):.8f}).",
            stacklevel=2,
        )
    else:
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
    silhouette_backend: str = "jax",
    silhouette_chunk_size: int = 1024,
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

    if silhouette_backend == "jax":
        from scib_metrics import silhouette_batch as jax_silhouette_batch
        from scib_metrics import silhouette_label as jax_silhouette_label

        embedding = np.asarray(work.obsm["X_emb"], dtype=np.float32)
        labels = work.obs["cell_type"].to_numpy()
        batches = work.obs["batch"].to_numpy()
        values["asw_label"] = _metric(
            "asw_label",
            lambda: jax_silhouette_label(
                embedding,
                labels,
                rescale=True,
                chunk_size=silhouette_chunk_size,
            ),
            errors,
            strict=strict,
        )
        values["asw_batch"] = _metric(
            "asw_batch",
            lambda: jax_silhouette_batch(
                embedding,
                labels,
                batches,
                rescale=True,
                chunk_size=silhouette_chunk_size,
            ),
            errors,
            strict=strict,
        )
    elif silhouette_backend == "scib":
        values["asw_label"] = _metric(
            "asw_label",
            lambda: scib_metrics.silhouette(work, label_key="cell_type", embed="X_emb"),
            errors,
            strict=strict,
        )
        values["asw_batch"] = _metric(
            "asw_batch",
            lambda: scib_metrics.silhouette_batch(
                work,
                batch_key="batch",
                label_key="cell_type",
                embed="X_emb",
                verbose=verbose,
            ),
            errors,
            strict=strict,
        )
    else:
        raise ValueError("silhouette_backend must be 'jax' or 'scib'")
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
        if "normalized" not in pre.layers:
            raise KeyError(
                "solution.layers is missing 'normalized', required for PCR and "
                "cell-cycle conservation"
            )
        # OpenProblems' partial H5AD reader maps layers/normalized to X before
        # calling these expression-aware scIB metrics.
        pre.X = pre.layers["normalized"].copy()
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
        "silhouette_backend": silhouette_backend,
        "silhouette_chunk_size": silhouette_chunk_size,
        "resolutions": list(OP_RESOLUTIONS),
        "embedding_key": embedding_key,
        "batch_key": batch_key,
        "label_key": label_key,
    }
    result.attrs["software_versions"] = _software_versions()
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
                "software_versions": result.attrs.get("software_versions", {}),
                "errors": result.attrs.get("errors", {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return output
