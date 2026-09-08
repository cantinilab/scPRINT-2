"""Pandora lung-atlas preprocessing and benchmark helpers."""

from __future__ import annotations

import importlib.metadata
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import requests
import scanpy as sc
import scipy.io
import scipy.sparse

BIOMART_URL = "https://useast.ensembl.org/biomart/martservice"
ORTHOLOGY_COLUMNS = {
    "Gene name": "mouse_gene_symbol",
    "Gene stable ID": "mouse_ensembl_gene_id",
    "Human gene stable ID": "human_ensembl_gene_id",
    "Human homology type": "human_homology_type",
    "Human orthology confidence [0 low, 1 high]": "human_orthology_confidence",
}
CELL_TYPE_ONTOLOGY = {
    "AT1": "CL:0002062",
    "ATI": "CL:0002062",
    "AT2": "CL:0002063",
    "ATII": "CL:0002063",
    "B cells": "CL:0000236",
    "Basal cells": "CL:0000646",
    "Ciliated cells": "CL:0000064",
    "Dendritic cells": "CL:0000451",
    "Dendrocytes": "CL:0000451",
    "Endothelial cells": "CL:0000115",
    "Fibroblasts": "CL:0000057",
    "Goblet cells": "CL:0000160",
    "Immune cells": "CL:0000738",
    "Macrophages": "CL:0000235",
    "Mast cells": "CL:0000097",
    "Mesothelial cells": "CL:0000077",
    "Monocytes": "CL:0000576",
    "NK cells": "CL:0000623",
    "Pericytes": "CL:0000669",
    "Secretory cells": "CL:0000151",
    "SMCs": "CL:0000192",
    "T cells": "CL:0000084",
}
DEFAULT_FILTERED_LABELS = ("unknown", "unmapped", "mix")

TARGET_FEATURE_SPACES = {
    "human_one2one": {
        "mapping_column": "human_ensembl_gene_id",
        "feature_organism_id": "NCBITaxon:9606",
        "target_feature_space": "high-confidence one-to-one human Ensembl genes",
        "no_mapping_error": "No source genes map to human one-to-one orthologs",
    },
    "mouse_homolog": {
        "mapping_column": "mouse_ensembl_gene_id",
        "feature_organism_id": "NCBITaxon:10090",
        "target_feature_space": "mouse homolog Ensembl genes",
        "no_mapping_error": "No source genes map to mouse Ensembl genes",
    },
}


@dataclass(frozen=True)
class PandoraBenchmarkConfig:
    """Python-facing configuration for a Pandora scIB benchmark."""

    expression: Path | str
    embedding: Path | str
    output: Path | str
    embedding_key: str = "scprint_emb"
    embedding_name: str | None = None
    additional_embeddings: tuple[tuple[str, Path | str, str], ...] = ()
    batch_key: str = "species"
    label_key: str = "cell_type_ontology_term_id"
    exclude_labels: tuple[str, ...] = DEFAULT_FILTERED_LABELS
    seed: int = 42
    n_jobs: int | None = None
    include_random: bool = False

    def __post_init__(self) -> None:
        names = [name for name, _, _ in self.additional_embeddings]
        if len(names) != len(set(names)):
            raise ValueError("Additional Pandora embedding names must be unique")
        if self.embedding_name in names:
            raise ValueError("Primary and additional embedding names must be unique")
        if self.n_jobs is not None and self.n_jobs < 1:
            raise ValueError("n_jobs must be positive")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("expression", "embedding", "output"):
            values[key] = str(values[key])
        values["additional_embeddings"] = [
            {"name": name, "path": str(path), "key": key}
            for name, path, key in self.additional_embeddings
        ]
        values["exclude_labels"] = list(values["exclude_labels"])
        return values


@dataclass
class PandoraBenchmarkResult:
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


def canonical_mouse_symbol(symbol: str) -> str:
    """Undo suffixes added by R's make.unique without altering dotted symbols."""
    head, separator, tail = str(symbol).rpartition(".")
    return head if separator and tail.isdigit() else str(symbol)


def summarize_pandora_annotations(annotation_audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize the validated Pandora cell-type table by species."""
    required = {"species", "cell_type", "cells"}
    missing = sorted(required - set(annotation_audit.columns))
    if missing:
        raise KeyError(f"Pandora annotation audit lacks columns: {missing}")
    table = annotation_audit.copy()
    table["cells"] = pd.to_numeric(table["cells"], errors="raise")
    return (
        table.groupby("species", as_index=False)
        .agg(cells=("cells", "sum"), cell_types=("cell_type", "nunique"))
        .sort_values("cells", ascending=False)
        .reset_index(drop=True)
    )


def summarize_pandora_orthology(orthologs: pd.DataFrame) -> pd.Series:
    """Return the one-to-one orthology checks used by the Pandora notebook."""
    required = {
        "mouse_gene_symbol",
        "mouse_ensembl_gene_id",
        "human_ensembl_gene_id",
        "human_homology_type",
        "human_orthology_confidence",
    }
    missing = sorted(required - set(orthologs.columns))
    if missing:
        raise KeyError(f"Pandora orthology table lacks columns: {missing}")
    high_confidence = orthologs["human_orthology_confidence"].astype(str).eq("1")
    one_to_one = orthologs["human_homology_type"].eq("ortholog_one2one")
    return pd.Series(
        {
            "rows": int(len(orthologs)),
            "mouse_symbols": int(orthologs["mouse_gene_symbol"].nunique()),
            "mouse_ensembl_genes": int(
                orthologs["mouse_ensembl_gene_id"].nunique()
            ),
            "human_ensembl_genes": int(
                orthologs["human_ensembl_gene_id"].nunique()
            ),
            "one_to_one_rows": int(one_to_one.sum()),
            "high_confidence_rows": int(high_confidence.sum()),
            "duplicate_mouse_symbols": int(
                orthologs["mouse_gene_symbol"].duplicated().sum()
            ),
        },
        name="Pandora orthology audit",
    )


def biomart_query() -> str:
    attributes = (
        "external_gene_name",
        "ensembl_gene_id",
        "hsapiens_homolog_ensembl_gene",
        "hsapiens_homolog_orthology_type",
        "hsapiens_homolog_orthology_confidence",
    )
    xml_attributes = "".join(f'<Attribute name="{name}"/>' for name in attributes)
    return (
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="1" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        f"{xml_attributes}</Dataset></Query>"
    )


def mouse_gene_query() -> str:
    attributes = ("external_gene_name", "ensembl_gene_id")
    xml_attributes = "".join(f'<Attribute name="{name}"/>' for name in attributes)
    return (
        '<Query virtualSchemaName="default" formatter="TSV" header="1" '
        'uniqueRows="1" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        f"{xml_attributes}</Dataset></Query>"
    )


def fetch_one_to_one_orthologs(url: str = BIOMART_URL) -> pd.DataFrame:
    response = requests.get(url, params={"query": biomart_query()}, timeout=120)
    response.raise_for_status()
    table = pd.read_csv(io.BytesIO(response.content), sep="\t", dtype=str)
    table = table.rename(columns=ORTHOLOGY_COLUMNS)
    missing = set(ORTHOLOGY_COLUMNS.values()) - set(table.columns)
    if missing:
        raise KeyError(f"BioMart response lacks columns: {sorted(missing)}")
    table = table.loc[
        (table["human_homology_type"] == "ortholog_one2one")
        & (table["human_orthology_confidence"] == "1")
    ].dropna(subset=["mouse_gene_symbol", "human_ensembl_gene_id"])
    ambiguous = table.groupby("mouse_gene_symbol")["human_ensembl_gene_id"].nunique()
    table = table.loc[~table["mouse_gene_symbol"].isin(ambiguous[ambiguous > 1].index)]
    return table.drop_duplicates("mouse_gene_symbol").sort_values("mouse_gene_symbol")


def fetch_mouse_gene_mapping(url: str = BIOMART_URL) -> pd.DataFrame:
    """Fetch unfiltered mouse symbols and their mouse Ensembl stable IDs."""
    response = requests.get(url, params={"query": mouse_gene_query()}, timeout=120)
    response.raise_for_status()
    table = pd.read_csv(io.BytesIO(response.content), sep="\t", dtype=str).rename(
        columns={
            "Gene name": "mouse_gene_symbol",
            "Gene stable ID": "mouse_ensembl_gene_id",
        }
    )
    required = {"mouse_gene_symbol", "mouse_ensembl_gene_id"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"BioMart response lacks columns: {sorted(missing)}")
    table = table.dropna(subset=sorted(required))
    table = table.loc[
        table["mouse_gene_symbol"].ne("")
        & table["mouse_ensembl_gene_id"].str.startswith("ENSMUSG")
    ]
    ambiguous = table.groupby("mouse_gene_symbol")["mouse_ensembl_gene_id"].nunique()
    table = table.loc[~table["mouse_gene_symbol"].isin(ambiguous[ambiguous > 1].index)]
    return table.drop_duplicates("mouse_gene_symbol").sort_values("mouse_gene_symbol")


def assemble_h5ad(
    matrix_path: Path,
    obs_path: Path,
    var_path: Path,
    output_path: Path,
    organism_id: str,
    species: str,
    source_count_sum: int | None = None,
    source_feature_count: int | None = None,
    mapped_source_feature_count: int | None = None,
    feature_organism_id: str = "NCBITaxon:9606",
) -> ad.AnnData:
    matrix = scipy.io.mmread(matrix_path).tocsr().transpose().tocsr()
    obs = pd.read_csv(obs_path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    var = pd.read_csv(var_path, sep="\t", index_col=0, dtype=str)
    return assemble_h5ad_from_objects(
        matrix,
        obs,
        var,
        output_path,
        organism_id,
        species,
        source_count_sum,
        source_feature_count,
        mapped_source_feature_count,
        feature_organism_id=feature_organism_id,
    )


def aggregate_source_genes(
    counts: scipy.sparse.csc_matrix,
    source_genes: pd.Index,
    mapping: pd.DataFrame,
    mapping_column: str,
) -> tuple[scipy.sparse.csr_matrix, pd.Index, int]:
    missing = {"mouse_gene_symbol", mapping_column} - set(mapping.columns)
    if missing:
        raise KeyError(f"Gene mapping lacks columns: {sorted(missing)}")
    mapping = mapping.dropna(subset=["mouse_gene_symbol", mapping_column])
    mapping = mapping.loc[
        (mapping["mouse_gene_symbol"].astype(str) != "")
        & (mapping[mapping_column].astype(str) != "")
    ]
    ambiguous = mapping.groupby("mouse_gene_symbol")[mapping_column].nunique()
    mapping = mapping.loc[
        ~mapping["mouse_gene_symbol"].isin(ambiguous[ambiguous > 1].index)
    ]
    lookup = mapping.drop_duplicates("mouse_gene_symbol").set_index(
        "mouse_gene_symbol"
    )[mapping_column]
    canonical = source_genes.map(canonical_mouse_symbol)
    targets = canonical.map(lookup)
    keep = pd.notna(targets)
    if not keep.any():
        return (
            scipy.sparse.csr_matrix((0, counts.shape[1]), dtype=np.float32),
            pd.Index([], dtype=str),
            0,
        )
    selected_targets = targets[keep]
    target_genes = pd.Index(sorted(selected_targets.unique()), dtype=str)
    aggregation = scipy.sparse.csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.float32),
            (target_genes.get_indexer(selected_targets), np.arange(int(keep.sum()))),
        ),
        shape=(len(target_genes), int(keep.sum())),
    )
    return (aggregation @ counts[keep]).tocsr(), target_genes, int(keep.sum())


def assemble_csc_h5ad(
    data_path: Path,
    indices_path: Path,
    indptr_path: Path,
    genes_path: Path,
    obs_path: Path,
    ortholog_path: Path,
    output_path: Path,
    organism_id: str,
    species: str,
    target_feature_space: str = "human_one2one",
) -> ad.AnnData:
    if target_feature_space not in TARGET_FEATURE_SPACES:
        raise ValueError(
            f"Unsupported target feature space {target_feature_space!r}; "
            f"expected one of {sorted(TARGET_FEATURE_SPACES)}"
        )
    obs = pd.read_csv(obs_path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    source_genes = pd.Index(genes_path.read_text().splitlines(), dtype=str)
    data = np.fromfile(data_path, dtype="<f8").astype(np.float32)
    indices = np.fromfile(indices_path, dtype="<i4")
    indptr = np.fromfile(indptr_path, dtype="<i4")
    counts = scipy.sparse.csc_matrix(
        (data, indices, indptr), shape=(len(source_genes), len(obs))
    )
    if "celltype" not in obs:
        raise KeyError("Pandora metadata has no 'celltype' annotation")

    target = TARGET_FEATURE_SPACES[target_feature_space]
    mapping_column = str(target["mapping_column"])
    mapping = pd.read_csv(ortholog_path, sep="\t", dtype=str)
    mapped_counts, mapped_genes, mapped_source_feature_count = aggregate_source_genes(
        counts, source_genes, mapping, mapping_column
    )
    if mapped_source_feature_count == 0:
        raise RuntimeError(str(target["no_mapping_error"]))
    var = pd.DataFrame(index=mapped_genes)
    return assemble_h5ad_from_objects(
        mapped_counts.transpose().tocsr(),
        obs,
        var,
        output_path,
        organism_id,
        species,
        source_count_sum=int(counts.sum()),
        source_feature_count=len(source_genes),
        mapped_source_feature_count=mapped_source_feature_count,
        feature_organism_id=str(target["feature_organism_id"]),
        target_feature_space=str(target["target_feature_space"]),
        mapping_column=mapping_column,
    )


def assemble_h5ad_from_objects(
    matrix,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    output_path: Path,
    organism_id: str,
    species: str,
    source_count_sum: int | None = None,
    source_feature_count: int | None = None,
    mapped_source_feature_count: int | None = None,
    feature_organism_id: str = "NCBITaxon:9606",
    source_feature_space: str = "mouse homolog gene symbols in Pandora RDS",
    target_feature_space: str = "high-confidence one-to-one human Ensembl genes",
    mapping_column: str | None = None,
) -> ad.AnnData:
    if matrix.shape != (len(obs), len(var)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match obs/var {(len(obs), len(var))}"
        )
    if "celltype" not in obs:
        raise KeyError("Pandora metadata has no 'celltype' annotation")
    obs = obs.copy()
    var = var.copy()
    obs["cell_type"] = obs["celltype"].astype(str)
    obs["cell_type_ontology_term_id"] = (
        obs["cell_type"].map(CELL_TYPE_ONTOLOGY).fillna("unknown")
    )
    obs["has_benchmark_label"] = obs["cell_type_ontology_term_id"] != "unknown"
    obs["source_organism_ontology_term_id"] = organism_id
    obs["organism_ontology_term_id"] = feature_organism_id
    obs["species"] = species
    obs["assay_ontology_term_id"] = "EFO:0030003"
    var.index.name = "ensembl_gene_id"
    var["ensembl_gene_id"] = var.index.astype(str)
    var["organism"] = feature_organism_id
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    mapping_metadata = {
        "source_feature_space": source_feature_space,
        "target_feature_space": target_feature_space,
        "target_feature_organism": feature_organism_id,
        "duplicate_policy": "sum counts after removing R make.unique numeric suffixes",
        "mapped_count_sum": int(matrix.sum()),
    }
    if mapping_column is not None:
        mapping_metadata["mapping_column"] = mapping_column
    if source_count_sum is not None:
        mapping_metadata["source_count_sum"] = source_count_sum
        mapping_metadata["count_retention_fraction"] = float(
            matrix.sum() / source_count_sum
        )
    if source_feature_count is not None:
        mapping_metadata["source_feature_count"] = source_feature_count
    if mapped_source_feature_count is not None:
        mapping_metadata["mapped_source_feature_count"] = mapped_source_feature_count
    result.uns["pandora_gene_mapping"] = mapping_metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output_path, compression="lzf")
    return result


def combine_h5ads(
    inputs: list[Path], output_path: Path, join: str = "inner"
) -> ad.AnnData:
    datasets = [ad.read_h5ad(path) for path in inputs]
    missing_feature_organism = [
        str(path)
        for path, dataset in zip(inputs, datasets)
        if "organism" not in dataset.var
    ]
    if missing_feature_organism:
        raise ValueError(
            "Pandora inputs must declare var['organism']; missing in: "
            + ", ".join(missing_feature_organism)
        )
    feature_organisms = {
        str(value)
        for dataset in datasets
        if "organism" in dataset.var
        for value in pd.unique(dataset.var["organism"].astype(str))
    }
    combined = ad.concat(
        datasets,
        join=join,
        merge="same",
        index_unique="-",
        label="pandora_dataset",
        keys=[path.stem for path in inputs],
    )
    combined.var["ensembl_gene_id"] = combined.var_names.astype(str)
    if len(feature_organisms) == 1:
        combined.var["organism"] = next(iter(feature_organisms))
    elif feature_organisms:
        raise ValueError(
            f"Cannot combine datasets with mixed feature organisms: {sorted(feature_organisms)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(output_path, compression="lzf")
    return combined


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _valid_label_mask(labels: pd.Series, excluded: set[str]) -> np.ndarray:
    normalized = labels.astype(str).str.strip().str.casefold()
    return (~normalized.isin(excluded) & normalized.ne("")).to_numpy()


def filter_h5ad_by_obs_labels(
    input_path: Path,
    output_path: Path,
    label_key: str = "cell_type_ontology_term_id",
    excluded_labels: tuple[str, ...] = DEFAULT_FILTERED_LABELS,
    expected_cells: int | None = None,
) -> ad.AnnData:
    dataset = ad.read_h5ad(input_path)
    if label_key not in dataset.obs:
        raise KeyError(f"Missing obs[{label_key!r}] in {input_path}")
    excluded = {value.strip().casefold() for value in excluded_labels}
    keep = _valid_label_mask(dataset.obs[label_key], excluded)
    if not keep.any():
        raise RuntimeError("No cells remain after filtering Pandora labels")
    filtered = dataset[keep].copy()
    if expected_cells is not None and filtered.n_obs != expected_cells:
        raise RuntimeError(
            f"Filtered Pandora atlas has {filtered.n_obs:,} cells; "
            f"expected {expected_cells:,}"
        )
    filtered.uns["pandora_cell_filter"] = {
        "source": str(input_path.resolve()),
        "label_key": label_key,
        "excluded_labels": sorted(excluded),
        "source_cells": int(dataset.n_obs),
        "filtered_cells": int(filtered.n_obs),
        "excluded_cells": int(dataset.n_obs - filtered.n_obs),
        "physical_filter": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_h5ad(output_path, compression="lzf")
    return filtered


def build_benchmark_data(
    expression: ad.AnnData,
    embedded: ad.AnnData,
    embedding_key: str,
    label_key: str,
    excluded_labels: set[str],
    seed: int,
) -> ad.AnnData:
    if not expression.obs_names.equals(embedded.obs_names):
        raise RuntimeError("Expression and embedding cell names/order differ")
    if embedding_key not in embedded.obsm:
        raise KeyError(f"Missing obsm[{embedding_key!r}] in embedding output")
    if label_key not in expression.obs:
        raise KeyError(f"Missing obs[{label_key!r}] in expression input")

    keep = _valid_label_mask(expression.obs[label_key], excluded_labels)
    if not keep.any():
        raise RuntimeError("No cells remain after excluding invalid labels")
    benchmark = expression[keep].copy()
    embedding = np.asarray(embedded.obsm[embedding_key][keep], dtype=np.float32)
    if embedding.shape[0] != benchmark.n_obs or not np.isfinite(embedding).all():
        raise RuntimeError("scPRINT embedding has invalid shape or non-finite values")
    benchmark.obsm[embedding_key] = embedding

    sc.pp.normalize_total(benchmark, target_sum=10_000)
    sc.pp.log1p(benchmark)
    n_components = min(50, benchmark.n_obs - 1, benchmark.n_vars - 1)
    if n_components < 2:
        raise RuntimeError("PCA requires at least three cells and three genes")
    sc.tl.pca(
        benchmark,
        n_comps=n_components,
        svd_solver="arpack",
        use_highly_variable=False,
        random_state=seed,
    )
    benchmark.obsm["X_pca"] = np.asarray(benchmark.obsm["X_pca"], dtype=np.float32)
    return benchmark


def attach_embedding(
    benchmark: ad.AnnData,
    expression: ad.AnnData,
    embedded: ad.AnnData,
    source_key: str,
    output_key: str,
    keep: np.ndarray,
) -> str:
    if source_key not in embedded.obsm:
        raise KeyError(f"Missing obsm[{source_key!r}] for {output_key}")
    if expression.obs_names.equals(embedded.obs_names):
        source_rows = np.arange(expression.n_obs)
        alignment = "exact_obs_names"
    else:
        identity = embedded.obs_names
        alignment = "obs_names_subset"
        recovered_key = "_transcriptformer_input_obs_name"
        if recovered_key in embedded.obs:
            recovered = pd.Index(embedded.obs[recovered_key].astype(str))
            if expression.obs_names.isin(recovered).all():
                identity = recovered
                alignment = recovered_key
            else:
                converted = pd.Index(
                    recovered.str.removesuffix(".human_one2one")
                    + ".mouse_homolog_ensmusg"
                )
                if expression.obs_names.isin(converted).all():
                    identity = converted
                    alignment = (
                        f"{recovered_key}:" ".human_one2one->.mouse_homolog_ensmusg"
                    )
        if not identity.is_unique:
            raise RuntimeError(f"Embedding {output_key} has duplicate cell names")
        source_rows = identity.get_indexer(expression.obs_names)
        if (source_rows < 0).any():
            missing = int((source_rows < 0).sum())
            raise RuntimeError(
                f"Embedding {output_key} is missing {missing} expression cells"
            )
    values = np.asarray(embedded.obsm[source_key][source_rows[keep]], dtype=np.float32)
    if values.shape[0] != benchmark.n_obs or not np.isfinite(values).all():
        raise RuntimeError(f"Embedding {output_key} has invalid shape or values")
    benchmark.obsm[output_key] = values
    return alignment


def run_pandora_benchmark(
    config: PandoraBenchmarkConfig,
) -> PandoraBenchmarkResult:
    """Run Pandora scIB directly from Python and return the result artifacts."""
    from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation

    expression_path = Path(config.expression)
    embedding_path = Path(config.embedding)
    output_path = Path(config.output)
    expression = ad.read_h5ad(expression_path)
    embedded = ad.read_h5ad(embedding_path)
    for column in (config.batch_key, config.label_key):
        if column not in expression.obs:
            raise KeyError(f"Missing obs[{column!r}] in expression input")

    excluded = {value.strip().casefold() for value in config.exclude_labels}
    benchmark_data = build_benchmark_data(
        expression,
        embedded,
        config.embedding_key,
        config.label_key,
        excluded,
        config.seed,
    )
    primary_name = config.embedding_name or config.embedding_key
    if primary_name != config.embedding_key:
        benchmark_data.obsm[primary_name] = benchmark_data.obsm.pop(
            config.embedding_key
        )
    keep = _valid_label_mask(expression.obs[config.label_key], excluded)
    embedding_sources = {
        primary_name: {
            "path": str(embedding_path.resolve()),
            "key": config.embedding_key,
        }
    }
    method_keys = [primary_name]
    for name, path_value, key in config.additional_embeddings:
        path = Path(path_value)
        if name in benchmark_data.obsm or name == "X_pca":
            raise ValueError(f"Duplicate or reserved embedding name: {name}")
        additional = ad.read_h5ad(path)
        alignment = attach_embedding(
            benchmark_data, expression, additional, key, name, keep
        )
        embedding_sources[name] = {
            "path": str(path.resolve()),
            "key": key,
            "cell_alignment": alignment,
        }
        method_keys.append(name)
    if config.include_random:
        rng = np.random.default_rng(config.seed)
        benchmark_data.obsm["random"] = rng.random(
            benchmark_data.obsm["X_pca"].shape, dtype=np.float32
        )
        method_keys.append("random")

    n_jobs = config.n_jobs or min(16, os.cpu_count() or 1)
    benchmark = Benchmarker(
        benchmark_data,
        batch_key=config.batch_key,
        label_key=config.label_key,
        embedding_obsm_keys=[*method_keys, "X_pca"],
        pre_integrated_embedding_obsm_key="X_pca",
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        n_jobs=n_jobs,
    )
    benchmark.benchmark()
    results = benchmark.get_results(min_max_scale=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\t" if output_path.suffix.casefold() == ".tsv" else ","
    results.to_csv(output_path, sep=separator)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "expression": str(expression_path.resolve()),
        "embeddings": embedding_sources,
        "batch_key": config.batch_key,
        "label_key": config.label_key,
        "excluded_labels": sorted(excluded),
        "source_cells": int(expression.n_obs),
        "scored_cells": int(benchmark_data.n_obs),
        "genes": int(expression.n_vars),
        "batches": int(benchmark_data.obs[config.batch_key].nunique()),
        "labels": int(benchmark_data.obs[config.label_key].nunique()),
        "methods": [*method_keys, "X_pca"],
        "output_separator": separator,
        "random_baseline": (
            {
                "distribution": "uniform_0_1",
                "dimension": int(benchmark_data.obsm["X_pca"].shape[1]),
                "seed": config.seed,
            }
            if config.include_random
            else None
        ),
        "pca": {
            "source": "expression.X",
            "normalization": "normalize_total_target_sum_10000_then_log1p",
            "uses_embeddings": False,
        },
        "min_max_scale": False,
        "seed": config.seed,
        "n_jobs": n_jobs,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("anndata", "scanpy", "scib-metrics", "scprint")
        },
    }
    if "pandora_cell_filter" in expression.uns:
        metadata["physical_filter"] = expression.uns["pandora_cell_filter"]
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default) + "\n"
    )
    complete_path = output_path.with_suffix(".COMPLETE")
    complete_path.write_text("COMPLETE\n")
    print(results.to_string())
    return PandoraBenchmarkResult(results, output_path, metadata_path, complete_path)
