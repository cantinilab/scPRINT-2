"""Four-species eye preprocessing for the cross-species benchmark.

The deposited GEO matrices contain integer counts, whereas the H5AD files used
in the original notebook contain only log-normalized expression.  This module
uses the H5AD files for expert annotations and the GEO files for expression.
It can then build either native per-species checkpoint blocks for scPRINT-2 or
a deterministic high-confidence mouse-ortholog union for shared baselines.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

MOUSE_TAXON = "NCBITaxon:10090"
HUMAN_TAXON = "NCBITaxon:9606"
EYE_CELL_TYPE_ONTOLOGY = {
    "B cell": "CL:0000236",
    # The exact eye-specific terms are newer than the small-v2 decoder.  Use
    # biologically compatible ancestors that are present in that checkpoint so
    # the labels can supervise fine-tuning instead of becoming ``unknown``.
    "Beam A": "CL:0000327",
    "Ciliary muscle": "CL:1000443",
    "Corneal endothelium": "CL:0000132",
    "Corneal epithelium": "CL:0000575",
    "Fibroblast": "CL:0000057",
    "JCT": "CL:0002320",
    "Macrophage": "CL:0000235",
    "Melanocyte": "CL:0000148",
    "NK/T cell": "CL:0000814",
    "Pericyte": "CL:0000669",
    "SChlemm's Canal": "CL:0000115",
    "Schwann cell": "CL:0002573",
}


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for one input artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_geo_cell_id(value: str) -> str:
    """Translate Seurat's terminal ``.1`` barcode suffix to GEO's ``-1``."""
    return re.sub(r"\.1$", "-1", str(value))


def extract_eye_annotations(
    source: ad.AnnData,
    *,
    species: str,
    source_organism_ontology_term_id: str,
    feature_organism_ontology_term_id: str | None = None,
) -> pd.DataFrame:
    """Extract and validate the expert annotations used by the eye benchmark."""
    label_key = "celltype" if "celltype" in source.obs else "cell_type"
    required = {label_key, "orig.ident"}
    missing = sorted(required - set(source.obs.columns))
    if missing:
        raise KeyError(f"Eye annotation source lacks columns: {missing}")

    labels = source.obs[label_key].astype("string")
    if labels.isna().any() or labels.str.strip().eq("").any():
        raise ValueError("Eye cell-type annotations contain missing values")
    unmapped = sorted(set(labels.astype(str)) - set(EYE_CELL_TYPE_ONTOLOGY))
    if unmapped:
        raise ValueError(f"Eye cell types lack ontology mappings: {unmapped}")

    result = pd.DataFrame(index=source.obs_names.astype(str))
    result.index.name = "source_obs_name"
    result["geo_cell_id"] = result.index.map(normalize_geo_cell_id)
    result["orig.ident"] = source.obs["orig.ident"].astype(str).values
    result["celltype"] = labels.astype(str).values
    result["cell_type"] = labels.astype(str).values
    result["cell_type_ontology_term_id"] = labels.map(
        EYE_CELL_TYPE_ONTOLOGY
    ).astype(str).values
    result["species"] = str(species)
    result["source_organism_ontology_term_id"] = str(
        source_organism_ontology_term_id
    )
    result["organism_ontology_term_id"] = str(
        feature_organism_ontology_term_id
        or source_organism_ontology_term_id
    )
    result["assay_ontology_term_id"] = "EFO:0030080"
    if not result.index.is_unique or not result["geo_cell_id"].is_unique:
        raise ValueError("Eye annotation cell identifiers are not unique")
    return result


def update_eye_annotation_contract(
    annotations: pd.DataFrame,
    *,
    species: str,
    source_organism_ontology_term_id: str,
    feature_organism_ontology_term_id: str | None = None,
) -> pd.DataFrame:
    """Apply current ontology and organism fields to an exported annotation table."""
    required = {"geo_cell_id", "orig.ident", "celltype"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise KeyError(f"Eye annotation table lacks columns: {missing}")
    result = annotations.copy()
    labels = result["celltype"].astype("string")
    if labels.isna().any() or labels.str.strip().eq("").any():
        raise ValueError("Eye cell-type annotations contain missing values")
    unmapped = sorted(set(labels.astype(str)) - set(EYE_CELL_TYPE_ONTOLOGY))
    if unmapped:
        raise ValueError(f"Eye cell types lack ontology mappings: {unmapped}")
    result["celltype"] = labels.astype(str).values
    result["cell_type"] = labels.astype(str).values
    result["cell_type_ontology_term_id"] = labels.map(
        EYE_CELL_TYPE_ONTOLOGY
    ).astype(str).values
    result["species"] = str(species)
    result["source_organism_ontology_term_id"] = str(
        source_organism_ontology_term_id
    )
    result["organism_ontology_term_id"] = str(
        feature_organism_ontology_term_id
        or source_organism_ontology_term_id
    )
    result["assay_ontology_term_id"] = "EFO:0030080"
    if not result.index.is_unique or not result["geo_cell_id"].astype(str).is_unique:
        raise ValueError("Eye annotation cell identifiers are not unique")
    return result


def strict_one_to_one_mapping(
    frames: Iterable[pd.DataFrame],
    *,
    source_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only gene pairs that are globally one-to-one across all inputs."""
    normalized = []
    input_rows = 0
    for frame in frames:
        missing = {source_col, target_col} - set(frame.columns)
        if missing:
            raise KeyError(f"Orthology table lacks columns: {sorted(missing)}")
        input_rows += len(frame)
        table = frame[[source_col, target_col]].copy()
        table.columns = ["source_gene", "mouse_gene"]
        normalized.append(table)
    if not normalized:
        raise ValueError("At least one orthology table is required")

    pairs = pd.concat(normalized, ignore_index=True).dropna()
    pairs = pairs.astype(str)
    pairs = pairs.loc[
        pairs["source_gene"].str.strip().ne("")
        & pairs["mouse_gene"].str.strip().ne("")
    ].drop_duplicates()
    source_counts = pairs.groupby("source_gene")["mouse_gene"].nunique()
    target_counts = pairs.groupby("mouse_gene")["source_gene"].nunique()
    source_conflicts = set(source_counts[source_counts > 1].index)
    target_conflicts = set(target_counts[target_counts > 1].index)
    strict = pairs.loc[
        ~pairs["source_gene"].isin(source_conflicts)
        & ~pairs["mouse_gene"].isin(target_conflicts)
    ].copy()
    strict = strict.sort_values(["mouse_gene", "source_gene"]).reset_index(drop=True)
    if not strict["source_gene"].is_unique or not strict["mouse_gene"].is_unique:
        raise RuntimeError("Strict orthology filtering did not produce one-to-one pairs")
    audit = {
        "input_rows": int(input_rows),
        "unique_pairs": int(len(pairs)),
        "conflicting_source_genes": int(len(source_conflicts)),
        "conflicting_mouse_genes": int(len(target_conflicts)),
        "retained_pairs": int(len(strict)),
    }
    return strict, audit


def resolve_orthology_conflicts(
    frames: Iterable[pd.DataFrame],
    *,
    source_col: str,
    target_col: str,
    homology_type_col: str = "homology_type",
    percent_identity_col: str = "source_homolog_percent_identity",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Choose a deterministic unique pair from high-confidence ortholog conflicts."""
    normalized = []
    input_rows = 0
    for frame in frames:
        required = {source_col, target_col, homology_type_col}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Orthology table lacks columns: {missing}")
        input_rows += len(frame)
        columns = [source_col, target_col, homology_type_col]
        if percent_identity_col in frame:
            columns.append(percent_identity_col)
        table = frame[columns].copy().rename(
            columns={
                source_col: "source_gene",
                target_col: "mouse_gene",
                homology_type_col: "homology_type",
                percent_identity_col: "percent_identity",
            }
        )
        if "percent_identity" not in table:
            table["percent_identity"] = np.nan
        normalized.append(table)
    if not normalized:
        raise ValueError("At least one orthology table is required")
    candidates = pd.concat(normalized, ignore_index=True).dropna(
        subset=["source_gene", "mouse_gene"]
    )
    candidates[["source_gene", "mouse_gene", "homology_type"]] = candidates[
        ["source_gene", "mouse_gene", "homology_type"]
    ].astype(str)
    candidates = candidates.loc[
        candidates["source_gene"].str.strip().ne("")
        & candidates["mouse_gene"].str.strip().ne("")
    ].copy()
    candidates["percent_identity"] = pd.to_numeric(
        candidates["percent_identity"], errors="coerce"
    ).fillna(-1.0)
    candidates["type_priority"] = candidates["homology_type"].map(
        {
            "ortholog_one2one": 0,
            "ortholog_one2many": 1,
            "ortholog_many2many": 2,
        }
    ).fillna(3)
    candidates = candidates.sort_values(
        [
            "type_priority",
            "percent_identity",
            "source_gene",
            "mouse_gene",
        ],
        ascending=[True, False, True, True],
    ).drop_duplicates(["source_gene", "mouse_gene"])
    used_sources: set[str] = set()
    used_targets: set[str] = set()
    retained_indices = []
    for index, row in candidates.iterrows():
        source_gene = str(row["source_gene"])
        mouse_gene = str(row["mouse_gene"])
        if source_gene in used_sources or mouse_gene in used_targets:
            continue
        used_sources.add(source_gene)
        used_targets.add(mouse_gene)
        retained_indices.append(index)
    retained = candidates.loc[retained_indices].sort_values(
        ["mouse_gene", "source_gene"]
    )
    if retained.empty:
        raise RuntimeError("Orthology conflict resolution retained no gene pairs")
    mapping = retained[["source_gene", "mouse_gene"]].reset_index(drop=True)
    audit = {
        "input_rows": int(input_rows),
        "unique_pairs": int(len(candidates)),
        "conflicting_source_genes": int(
            (candidates.groupby("source_gene")["mouse_gene"].nunique() > 1).sum()
        ),
        "conflicting_mouse_genes": int(
            (candidates.groupby("mouse_gene")["source_gene"].nunique() > 1).sum()
        ),
        "retained_pairs": int(len(mapping)),
        "retained_one2one": int(
            retained["homology_type"].eq("ortholog_one2one").sum()
        ),
        "retained_non_one2one": int(
            (~retained["homology_type"].eq("ortholog_one2one")).sum()
        ),
        "mode": "high_confidence_deterministic_unique_orthologs",
    }
    return mapping, audit


def direct_symbol_to_ensembl_mapping(
    frame: pd.DataFrame,
    *,
    observed_genes: Sequence[str],
    symbol_col: str,
    ensembl_col: str,
    normalize_terminal_np_suffixes: bool = False,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Map HGNC-like source symbols to one deterministic human Ensembl ID.

    The registry behavior intentionally mirrors ``scdataloader.Preprocessor``:
    Ensembl IDs are sorted and duplicated symbols retain the first stable ID.
    Terminal ``-n/-p`` and ``_n/_p`` aliases are only considered after an exact
    symbol miss; multiple source rows that reach one Ensembl target are retained
    so their integer counts can be summed during GEO import.
    """
    required = {symbol_col, ensembl_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Human gene registry lacks columns: {missing}")

    registry = frame[[symbol_col, ensembl_col]].copy().dropna()
    registry = registry.astype(str)
    registry = registry.loc[
        registry[symbol_col].str.strip().ne("")
        & registry[ensembl_col].str.strip().ne("")
    ].drop_duplicates()
    if registry.empty:
        raise ValueError("Human gene registry contains no usable symbol/Ensembl pairs")
    symbol_target_counts = registry.groupby(symbol_col)[ensembl_col].nunique()
    lookup = (
        registry.sort_values([ensembl_col, symbol_col])
        .drop_duplicates(symbol_col)
        .set_index(symbol_col)[ensembl_col]
    )

    records: list[tuple[str, str]] = []
    exact_hits = 0
    suffix_hits = 0
    for source_gene in map(str, observed_genes):
        matched_symbol = source_gene
        if matched_symbol in lookup.index:
            exact_hits += 1
        elif normalize_terminal_np_suffixes:
            matched_symbol = re.sub(r"[-_][np]$", "", source_gene)
            if matched_symbol == source_gene or matched_symbol not in lookup.index:
                continue
            suffix_hits += 1
        else:
            continue
        records.append((source_gene, str(lookup.loc[matched_symbol])))

    mapping = pd.DataFrame(records, columns=["source_gene", "target_gene"])
    if mapping.empty:
        raise ValueError("No observed HGNC-like symbols map to human Ensembl IDs")
    if not mapping["source_gene"].is_unique:
        raise RuntimeError("Observed source genes are duplicated in direct symbol mapping")
    target_counts = mapping["target_gene"].value_counts()
    audit: dict[str, int | str] = {
        "mode": "direct_human_symbol_to_ensembl",
        "input_rows": int(len(frame)),
        "unique_pairs": int(len(registry)),
        "registry_symbols": int(registry[symbol_col].nunique()),
        "ambiguous_registry_symbols": int((symbol_target_counts > 1).sum()),
        "observed_gene_rows": int(len(observed_genes)),
        "exact_symbol_hits": int(exact_hits),
        "terminal_np_suffix_hits": int(suffix_hits),
        "retained_pairs": int(len(mapping)),
        "retained_targets": int(mapping["target_gene"].nunique()),
        "collapsed_target_genes": int((target_counts > 1).sum()),
    }
    return mapping, audit


def resolve_source_orthology_targets(
    frames: Iterable[pd.DataFrame],
    *,
    source_col: str,
    target_col: str,
    homology_type_col: str = "homology_type",
    percent_identity_col: str = "source_homolog_percent_identity",
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Choose one deterministic ortholog target per source and allow aggregation.

    Unlike :func:`resolve_orthology_conflicts`, distinct source genes may map to
    the same target Ensembl ID. Their counts are aggregated instead of dropping
    biologically valid source rows merely to manufacture a one-to-one table.
    """
    normalized = []
    input_rows = 0
    for frame in frames:
        required = {source_col, target_col, homology_type_col}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Orthology table lacks columns: {missing}")
        input_rows += len(frame)
        columns = [source_col, target_col, homology_type_col]
        if percent_identity_col in frame:
            columns.append(percent_identity_col)
        table = frame[columns].copy().rename(
            columns={
                source_col: "source_gene",
                target_col: "target_gene",
                homology_type_col: "homology_type",
                percent_identity_col: "percent_identity",
            }
        )
        if "percent_identity" not in table:
            table["percent_identity"] = np.nan
        normalized.append(table)
    if not normalized:
        raise ValueError("At least one orthology table is required")

    candidates = pd.concat(normalized, ignore_index=True).dropna(
        subset=["source_gene", "target_gene"]
    )
    candidates[["source_gene", "target_gene", "homology_type"]] = candidates[
        ["source_gene", "target_gene", "homology_type"]
    ].astype(str)
    candidates = candidates.loc[
        candidates["source_gene"].str.strip().ne("")
        & candidates["target_gene"].str.strip().ne("")
    ].copy()
    candidates["percent_identity"] = pd.to_numeric(
        candidates["percent_identity"], errors="coerce"
    ).fillna(-1.0)
    candidates["type_priority"] = candidates["homology_type"].map(
        {
            "ortholog_one2one": 0,
            "ortholog_one2many": 1,
            "ortholog_many2many": 2,
        }
    ).fillna(3)
    candidates = candidates.sort_values(
        ["type_priority", "percent_identity", "source_gene", "target_gene"],
        ascending=[True, False, True, True],
    ).drop_duplicates(["source_gene", "target_gene"])
    source_conflicts = int(
        (candidates.groupby("source_gene")["target_gene"].nunique() > 1).sum()
    )
    retained = candidates.drop_duplicates("source_gene").copy()
    if retained.empty:
        raise RuntimeError("Orthology conflict resolution retained no gene pairs")
    target_counts = retained["target_gene"].value_counts()
    mapping = retained[["source_gene", "target_gene"]].sort_values(
        ["target_gene", "source_gene"]
    ).reset_index(drop=True)
    audit: dict[str, int | str] = {
        "mode": "high_confidence_source_unique_target_aggregated_orthologs",
        "input_rows": int(input_rows),
        "unique_pairs": int(len(candidates)),
        "conflicting_source_genes": source_conflicts,
        "conflicting_target_genes": int((target_counts > 1).sum()),
        "retained_pairs": int(len(mapping)),
        "retained_targets": int(mapping["target_gene"].nunique()),
        "retained_one2one": int(
            retained["homology_type"].eq("ortholog_one2one").sum()
        ),
        "retained_non_one2one": int(
            (~retained["homology_type"].eq("ortholog_one2one")).sum()
        ),
    }
    return mapping, audit


def merge_preferred_target_mappings(
    preferred: pd.DataFrame,
    fallback: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    """Merge source-to-target mappings while preserving direct symbol matches.

    A GEO feature that already matches a human symbol keeps that direct ENSG
    assignment.  Orthology is used only for remaining source identifiers, such
    as pig ``ENSSSCG`` IDs.  Multiple source rows may still converge on one
    human target and are aggregated by :func:`load_geo_count_subset`.
    """
    required = {"source_gene", "target_gene"}
    for name, mapping in (("preferred", preferred), ("fallback", fallback)):
        missing = sorted(required - set(mapping.columns))
        if missing:
            raise KeyError(f"{name.capitalize()} mapping lacks columns: {missing}")
        if not mapping["source_gene"].is_unique:
            raise ValueError(f"{name.capitalize()} mapping source genes must be unique")

    preferred_clean = preferred[["source_gene", "target_gene"]].copy()
    fallback_clean = fallback[["source_gene", "target_gene"]].copy()
    preferred_sources = set(preferred_clean["source_gene"].astype(str))
    fallback_clean = fallback_clean.loc[
        ~fallback_clean["source_gene"].astype(str).isin(preferred_sources)
    ]
    mapping = pd.concat([preferred_clean, fallback_clean], ignore_index=True)
    mapping = mapping.astype(str).sort_values(
        ["target_gene", "source_gene"]
    ).reset_index(drop=True)
    if mapping.empty or not mapping["source_gene"].is_unique:
        raise RuntimeError("Preferred/fallback mapping merge is not source-unique")
    target_counts = mapping["target_gene"].value_counts()
    audit: dict[str, int | str] = {
        "mode": "direct_human_symbol_preferred_with_orthology_fallback",
        "preferred_pairs": int(len(preferred_clean)),
        "fallback_input_pairs": int(len(fallback)),
        "fallback_added_pairs": int(len(fallback_clean)),
        "retained_pairs": int(len(mapping)),
        "retained_targets": int(mapping["target_gene"].nunique()),
        "collapsed_target_genes": int((target_counts > 1).sum()),
    }
    return mapping, audit


def identity_gene_mapping(genes: Sequence[str]) -> pd.DataFrame:
    """Construct the mouse-to-mouse identity mapping after uniqueness checks."""
    values = pd.Index(genes, dtype=str)
    if not values.is_unique:
        duplicates = values[values.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Mouse GEO gene symbols are duplicated: {duplicates}")
    return pd.DataFrame({"source_gene": values, "mouse_gene": values})


def expand_observed_gene_aliases(
    frames: Iterable[pd.DataFrame],
    *,
    source_columns: Sequence[str],
    observed_genes: Sequence[str],
    expanded_source_column: str = "_observed_source_gene",
) -> list[pd.DataFrame]:
    """Expand symbol/Ensembl aliases, retaining only identifiers present in GEO."""
    columns = [str(column) for column in source_columns]
    if not columns:
        raise ValueError("At least one source gene identifier column is required")
    observed = set(map(str, observed_genes))
    if not observed:
        raise ValueError("The GEO matrix contains no observed gene identifiers")

    expanded: list[pd.DataFrame] = []
    for frame in frames:
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise KeyError(f"Gene mapping lacks alias columns: {missing}")
        for column in columns:
            table = frame.copy()
            table[expanded_source_column] = table[column].astype(str)
            table = table.loc[table[expanded_source_column].isin(observed)].copy()
            if not table.empty:
                expanded.append(table)
    if not expanded:
        raise ValueError("No mapping aliases occur in the GEO gene identifiers")
    return expanded


def load_geo_count_subset(
    count_csv_gz: Path | str,
    annotations: pd.DataFrame,
    mapping: pd.DataFrame | None,
    *,
    chunksize: int = 64,
    aggregate_duplicate_targets: bool = False,
    target_id_type: str = "symbol",
) -> ad.AnnData:
    """Load annotated GEO cells and mapped genes without densifying the matrix."""
    path = Path(count_csv_gz)
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    required_annotations = {
        "geo_cell_id",
        "celltype",
        "species",
        "source_organism_ontology_term_id",
        "organism_ontology_term_id",
    }
    missing_annotations = sorted(required_annotations - set(annotations.columns))
    if missing_annotations:
        raise KeyError(f"Eye annotations lack columns: {missing_annotations}")
    if not annotations.index.is_unique or not annotations["geo_cell_id"].is_unique:
        raise ValueError("Eye annotation cell identifiers are not unique")

    header = pd.read_csv(path, nrows=0)
    gene_column = str(header.columns[0])
    geo_cells = annotations["geo_cell_id"].astype(str).tolist()
    missing_cells = sorted(set(geo_cells) - set(header.columns))
    if missing_cells:
        raise ValueError(
            f"{len(missing_cells)} annotated cells are absent from {path.name}; "
            f"examples: {missing_cells[:5]}"
        )

    lookup = None
    if mapping is not None:
        target_column = (
            "target_gene" if "target_gene" in mapping else "mouse_gene"
        )
        required_mapping = {"source_gene", target_column}
        missing_mapping = sorted(required_mapping - set(mapping.columns))
        if missing_mapping:
            raise KeyError(f"Gene mapping lacks columns: {missing_mapping}")
        if not mapping["source_gene"].is_unique:
            raise ValueError("Gene mapping source identifiers must be unique")
        if not aggregate_duplicate_targets and not mapping[target_column].is_unique:
            raise ValueError("Gene mapping must be strict one-to-one")
        lookup = mapping.set_index("source_gene")[target_column]

    matrix_parts: list[sp.csr_matrix] = []
    mapped_genes: list[str] = []
    source_genes_seen: set[str] = set()
    source_gene_rows = 0
    usecols = [gene_column, *geo_cells]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        source_genes = chunk[gene_column].astype(str)
        source_gene_rows += len(source_genes)
        if lookup is None:
            keep = pd.Series(True, index=chunk.index)
            targets = source_genes
        else:
            keep = source_genes.isin(lookup.index)
            targets = source_genes.loc[keep].map(lookup)
        if not keep.any():
            continue
        selected_sources = source_genes.loc[keep].tolist()
        repeated = source_genes_seen.intersection(selected_sources)
        if repeated:
            raise ValueError(
                "Mapped source genes occur more than once in the GEO matrix: "
                f"{sorted(repeated)[:10]}"
            )
        source_genes_seen.update(selected_sources)

        values = chunk.loc[keep, geo_cells].to_numpy()
        if not np.issubdtype(values.dtype, np.number):
            values = np.asarray(pd.DataFrame(values).apply(pd.to_numeric), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("GEO expression contains non-finite or negative values")
        if not np.equal(values, np.rint(values)).all():
            raise ValueError("GEO expression is not an integer count matrix")
        matrix_parts.append(sp.csr_matrix(values.astype(np.int32, copy=False)))
        mapped_genes.extend(targets.astype(str).tolist())

    if not matrix_parts:
        raise ValueError(f"No mapped genes were loaded from {path}")
    if not aggregate_duplicate_targets and len(mapped_genes) != len(set(mapped_genes)):
        duplicates = pd.Index(mapped_genes)[pd.Index(mapped_genes).duplicated()]
        raise ValueError(f"Mapped target genes are duplicated: {duplicates[:10].tolist()}")

    gene_by_cell = sp.vstack(matrix_parts, format="csr")
    if aggregate_duplicate_targets:
        target_genes = pd.Index(sorted(set(mapped_genes)), name="gene")
        target_positions = pd.Series(
            np.arange(len(target_genes), dtype=np.int64), index=target_genes
        )
        source_positions = target_positions.loc[mapped_genes].to_numpy(dtype=np.int64)
        source_to_target = sp.csr_matrix(
            (
                np.ones(len(mapped_genes), dtype=np.int32),
                (np.arange(len(mapped_genes), dtype=np.int64), source_positions),
            ),
            shape=(len(mapped_genes), len(target_genes)),
        )
        gene_by_cell = source_to_target.transpose().dot(gene_by_cell).tocsr()
        mapped_genes = target_genes.astype(str).tolist()
    counts = gene_by_cell.transpose().tocsr()
    obs = annotations.copy()
    obs.index = obs.index.astype(str)
    feature_organisms = annotations["organism_ontology_term_id"].astype(str).unique()
    if len(feature_organisms) != 1:
        raise ValueError("One GEO source matrix must contain exactly one organism")
    var = pd.DataFrame(index=pd.Index(mapped_genes, name="gene"))
    if target_id_type == "symbol":
        var["symbol"] = var.index.astype(str)
    elif target_id_type == "ensembl":
        var["symbol"] = ""
        var["ensembl_gene_id"] = var.index.astype(str)
    else:
        raise ValueError("target_id_type must be 'symbol' or 'ensembl'")
    var["organism"] = feature_organisms[0]
    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.uns["eye_geo_import"] = {
        "source": str(path.resolve()),
        "source_sha256": file_sha256(path),
        "source_gene_rows": int(source_gene_rows),
        "mapped_gene_rows": int(result.n_vars),
        "mapped_source_gene_rows": int(len(source_genes_seen)),
        "aggregated_target_gene_rows": int(len(source_genes_seen) - result.n_vars),
        "annotated_cells": int(result.n_obs),
        "count_sum": int(result.X.sum()),
        "integer_counts": True,
    }
    return result


def combine_eye_species(
    datasets: Sequence[ad.AnnData],
    *,
    min_shared_genes: int = 1,
) -> ad.AnnData:
    """Combine species in their shared strict mouse-ortholog feature space."""
    if len(datasets) < 2:
        raise ValueError("At least two eye species datasets are required")
    common = set(datasets[0].var_names.astype(str))
    for dataset in datasets[1:]:
        common.intersection_update(dataset.var_names.astype(str))
    common_genes = pd.Index(sorted(common), name="mouse_gene_symbol")
    if len(common_genes) < min_shared_genes:
        raise ValueError(
            f"Only {len(common_genes)} shared mouse genes; expected at least "
            f"{min_shared_genes}"
        )

    obs = pd.concat([dataset.obs.copy() for dataset in datasets], axis=0)
    if not obs.index.is_unique:
        duplicates = obs.index[obs.index.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Cell identifiers overlap across species: {duplicates}")
    counts = sp.vstack(
        [dataset[:, common_genes].X for dataset in datasets], format="csr"
    )
    if not np.equal(counts.data, np.rint(counts.data)).all() or (counts.data < 0).any():
        raise ValueError("Combined eye matrix is not non-negative integer counts")
    var = pd.DataFrame(index=common_genes)
    var["symbol"] = common_genes.astype(str)
    var["organism"] = MOUSE_TAXON
    result = ad.AnnData(X=counts.astype(np.int32), obs=obs, var=var)
    result.uns["eye_cross_species"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "species": sorted(result.obs["species"].astype(str).unique().tolist()),
        "cells_by_species": {
            str(key): int(value)
            for key, value in result.obs["species"].value_counts().items()
        },
        "shared_mouse_gene_symbols": int(result.n_vars),
        "count_sum": int(result.X.sum()),
        "integer_counts": True,
        "feature_organism_ontology_term_id": MOUSE_TAXON,
        "scib_batch_key": "species",
        "scib_label_key": "celltype",
    }
    return result


def combine_eye_ortholog_species(
    datasets: Sequence[ad.AnnData],
    *,
    min_genes_per_species: int = 15_000,
    min_common_genes: int = 10_000,
) -> ad.AnnData:
    """Combine per-species mouse orthologs without taking a four-way intersection."""
    if len(datasets) < 2:
        raise ValueError("At least two eye species datasets are required")
    gene_sets = [set(dataset.var_names.astype(str)) for dataset in datasets]
    too_small = [
        (str(dataset.obs["species"].astype(str).iloc[0]), dataset.n_vars)
        for dataset in datasets
        if dataset.n_vars < min_genes_per_species
    ]
    if too_small:
        raise ValueError(
            "Eye species have fewer mapped orthologs than required: "
            + ", ".join(f"{species}={genes}" for species, genes in too_small)
        )
    common = set.intersection(*gene_sets)
    if len(common) < min_common_genes:
        raise ValueError(
            f"Only {len(common)} common mouse orthologs; expected at least "
            f"{min_common_genes}"
        )
    union = pd.Index(sorted(set.union(*gene_sets)), name="mouse_gene_symbol")
    union_positions = pd.Series(np.arange(len(union), dtype=np.int64), index=union)
    matrices = []
    for dataset in datasets:
        matrix = sp.csr_matrix(dataset.X).tocoo()
        columns = union_positions.loc[
            dataset.var_names.astype(str)
        ].to_numpy(dtype=np.int64)
        matrices.append(
            sp.csr_matrix(
                (matrix.data, (matrix.row, columns[matrix.col])),
                shape=(dataset.n_obs, len(union)),
                dtype=np.int32,
            )
        )
    counts = sp.vstack(matrices, format="csr")
    obs = pd.concat([dataset.obs.copy() for dataset in datasets], axis=0)
    if not obs.index.is_unique:
        raise ValueError("Cell identifiers overlap across ortholog eye species")
    var = pd.DataFrame(index=union)
    var["symbol"] = union.astype(str)
    var["organism"] = MOUSE_TAXON
    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.uns["eye_cross_species"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "feature_space": "union_of_per_species_mouse_orthologs",
        "species": sorted(result.obs["species"].astype(str).unique().tolist()),
        "cells_by_species": {
            str(key): int(value)
            for key, value in result.obs["species"].value_counts().items()
        },
        "mapped_mouse_orthologs_by_species": {
            str(dataset.obs["species"].astype(str).iloc[0]): int(dataset.n_vars)
            for dataset in datasets
        },
        "common_mouse_orthologs": int(len(common)),
        "union_mouse_orthologs": int(len(union)),
        "count_sum": int(counts.sum()),
        "integer_counts": True,
        "feature_organism_ontology_term_id": MOUSE_TAXON,
        "scib_batch_key": "species",
        "scib_label_key": "celltype",
    }
    return result


def combine_eye_target_species(
    datasets: Sequence[ad.AnnData],
    *,
    target_organism_ontology_term_id: str,
    feature_space: str,
    min_genes_per_species: int = 14_000,
) -> ad.AnnData:
    """Combine target-species Ensembl features by union, never intersection."""
    if len(datasets) < 2:
        raise ValueError("At least two target-space eye datasets are required")
    gene_sets = []
    mapped_by_species: dict[str, int] = {}
    expressed_by_species: dict[str, int] = {}
    for dataset in datasets:
        if not dataset.var_names.is_unique:
            raise ValueError("Target-space eye gene identifiers must be unique")
        organisms = dataset.obs["organism_ontology_term_id"].astype(str).unique()
        if organisms.tolist() != [target_organism_ontology_term_id]:
            raise ValueError(
                "Target-space feature organism mismatch: "
                f"{organisms.tolist()} != {target_organism_ontology_term_id}"
            )
        species = str(dataset.obs["species"].astype(str).iloc[0])
        if dataset.n_vars < min_genes_per_species:
            raise ValueError(
                f"Only {dataset.n_vars} target genes for {species}; expected at least "
                f"{min_genes_per_species}"
            )
        gene_sets.append(set(dataset.var_names.astype(str)))
        mapped_by_species[species] = int(dataset.n_vars)
        expressed_by_species[species] = int(
            np.asarray(sp.csr_matrix(dataset.X).getnnz(axis=0) > 0).sum()
        )

    common = set.intersection(*gene_sets)
    union = pd.Index(sorted(set.union(*gene_sets)), name="ensembl_gene_id")
    union_positions = pd.Series(np.arange(len(union), dtype=np.int64), index=union)
    matrices = []
    for dataset in datasets:
        matrix = sp.csr_matrix(dataset.X).tocoo()
        columns = union_positions.loc[
            dataset.var_names.astype(str)
        ].to_numpy(dtype=np.int64)
        matrices.append(
            sp.csr_matrix(
                (matrix.data, (matrix.row, columns[matrix.col])),
                shape=(dataset.n_obs, len(union)),
                dtype=np.int32,
            )
        )
    counts = sp.vstack(matrices, format="csr")
    obs = pd.concat([dataset.obs.copy() for dataset in datasets], axis=0)
    if not obs.index.is_unique:
        raise ValueError("Cell identifiers overlap across target-space eye species")
    var = pd.DataFrame(index=union)
    var["ensembl_gene_id"] = union.astype(str)
    var["symbol"] = ""
    var["organism"] = target_organism_ontology_term_id
    result = ad.AnnData(X=counts, obs=obs, var=var)
    result.uns["eye_cross_species"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "feature_space": feature_space,
        "species": sorted(result.obs["species"].astype(str).unique().tolist()),
        "cells_by_species": {
            str(key): int(value)
            for key, value in result.obs["species"].value_counts().items()
        },
        "mapped_target_genes_by_species": mapped_by_species,
        "expressed_target_genes_by_species": expressed_by_species,
        "common_target_genes_audit_only": int(len(common)),
        "union_target_genes": int(len(union)),
        "combination_rule": "union_zero_fill_no_intersection_filter",
        "count_sum": int(counts.sum()),
        "integer_counts": True,
        "feature_organism_ontology_term_id": target_organism_ontology_term_id,
        "scib_batch_key": "species",
        "scib_label_key": "celltype",
    }
    return result


def checkpoint_gene_vocabularies(
    checkpoint: Path | str,
) -> tuple[dict[str, list[str]], list[str]]:
    """Read ordered per-organism gene vocabularies from a scPRINT checkpoint."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyper_parameters = payload.get("hyper_parameters", {})
    genes = hyper_parameters.get("genes")
    organisms = [str(value) for value in hyper_parameters.get("organisms", [])]
    if not isinstance(genes, dict) or not organisms:
        raise ValueError("Checkpoint lacks ordered per-organism gene vocabularies")
    missing = [organism for organism in organisms if organism not in genes]
    if missing:
        raise KeyError(f"Checkpoint lacks genes for organisms: {missing}")
    vocabularies = {
        organism: [str(gene) for gene in genes[organism]]
        for organism in organisms
    }
    for organism, values in vocabularies.items():
        if len(values) != len(set(values)):
            raise ValueError(f"Checkpoint genes are duplicated for {organism}")
    return vocabularies, organisms


def align_eye_to_checkpoint_genes(
    source: ad.AnnData | Path | str,
    genes_by_organism: dict[str, Sequence[str]],
    *,
    min_checkpoint_genes: int = 15_000,
    min_input_gene_overlap: int = 15_000,
) -> ad.AnnData:
    """Align one native-species dataset to its complete checkpoint vocabulary."""
    data = ad.read_h5ad(source) if isinstance(source, (Path, str)) else source.copy()
    organisms = data.obs["organism_ontology_term_id"].astype(str).unique().tolist()
    if len(organisms) != 1:
        raise ValueError(f"Expected one organism before checkpoint alignment: {organisms}")
    organism = organisms[0]
    if organism not in genes_by_organism:
        raise KeyError(f"Checkpoint does not support {organism}")
    expected = pd.Index(
        [str(gene) for gene in genes_by_organism[organism]],
        name="ensembl_gene_id",
    )
    if not expected.is_unique:
        raise ValueError(f"Checkpoint genes are duplicated for {organism}")
    if len(expected) < min_checkpoint_genes:
        raise ValueError(
            f"Checkpoint has only {len(expected)} genes for {organism}; "
            f"expected at least {min_checkpoint_genes}"
        )
    if not data.var_names.is_unique:
        raise ValueError("Preprocessed eye gene identifiers are not unique")

    expected_positions = pd.Series(
        np.arange(len(expected), dtype=np.int64), index=expected
    )
    observed = pd.Index(data.var_names.astype(str))
    overlap_mask = observed.isin(expected)
    overlap = observed[overlap_mask]
    if overlap.empty:
        raise RuntimeError(f"No preprocessed genes overlap {organism} checkpoint genes")
    if len(overlap) < min_input_gene_overlap:
        raise ValueError(
            f"Only {len(overlap)} input genes overlap the {organism} checkpoint; "
            f"expected at least {min_input_gene_overlap}"
        )
    source_counts = data.X
    if not sp.issparse(source_counts):
        source_counts = sp.csr_matrix(source_counts)
    if (source_counts.data < 0).any() or not np.equal(
        source_counts.data, np.rint(source_counts.data)
    ).all():
        raise ValueError("Checkpoint alignment requires non-negative integer counts")
    selected = source_counts[:, overlap_mask].tocoo()
    target_positions = expected_positions.loc[overlap].to_numpy(dtype=np.int64)
    aligned = sp.csr_matrix(
        (
            selected.data,
            (selected.row, target_positions[selected.col]),
        ),
        shape=(data.n_obs, len(expected)),
        dtype=np.int32,
    )

    source_var = data.var.copy()
    source_var.index = observed
    var = source_var.reindex(expected).copy()
    var["organism"] = organism
    var["ensembl_gene_id"] = expected.astype(str)
    if "symbol" not in var:
        var["symbol"] = ""
    result = ad.AnnData(
        X=aligned,
        obs=data.obs.copy(),
        var=var,
        uns=data.uns.copy(),
    )
    result.uns["eye_native_checkpoint_alignment"] = {
        "organism": organism,
        "checkpoint_genes": int(len(expected)),
        "preprocessed_genes": int(data.n_vars),
        "preprocessed_gene_overlap": int(len(overlap)),
        "expressed_checkpoint_genes": int(
            np.asarray(aligned.getnnz(axis=0) > 0).sum()
        ),
        "count_sum": int(aligned.sum()),
        "integer_counts": True,
        "zero_filled_checkpoint_genes": int(len(expected) - len(overlap)),
    }
    return result


def combine_eye_native_species(
    datasets: Sequence[ad.AnnData],
    *,
    organism_order: Sequence[str],
    min_genes_per_species: int = 15_000,
) -> ad.AnnData:
    """Combine native checkpoint vocabularies as a sparse block-diagonal matrix."""
    if len(datasets) < 2:
        raise ValueError("At least two native eye species datasets are required")
    by_organism: dict[str, ad.AnnData] = {}
    for dataset in datasets:
        organisms = (
            dataset.obs["organism_ontology_term_id"].astype(str).unique().tolist()
        )
        if len(organisms) != 1:
            raise ValueError(f"Expected one organism per dataset: {organisms}")
        organism = organisms[0]
        if organism in by_organism:
            raise ValueError(f"Duplicate native eye dataset for {organism}")
        if dataset.n_vars < min_genes_per_species:
            raise ValueError(
                f"Only {dataset.n_vars} genes for {organism}; expected at least "
                f"{min_genes_per_species}"
            )
        if "organism" not in dataset.var or set(
            dataset.var["organism"].astype(str)
        ) != {organism}:
            raise ValueError(f"Gene organism metadata does not match {organism}")
        by_organism[organism] = dataset

    ordered = [by_organism[value] for value in organism_order if value in by_organism]
    if len(ordered) != len(datasets):
        missing = sorted(set(by_organism) - set(organism_order))
        raise ValueError(f"Organism order is incomplete: {missing}")
    obs = pd.concat([dataset.obs.copy() for dataset in ordered], axis=0)
    var = pd.concat([dataset.var.copy() for dataset in ordered], axis=0)
    if not obs.index.is_unique:
        raise ValueError("Cell identifiers overlap across native eye species")
    if not var.index.is_unique:
        raise ValueError("Checkpoint gene identifiers overlap across eye species")
    counts = sp.block_diag(
        [sp.csr_matrix(dataset.X) for dataset in ordered], format="csr"
    ).astype(np.int32)
    if (counts.data < 0).any() or not np.equal(counts.data, np.rint(counts.data)).all():
        raise ValueError("Combined native eye matrix is not non-negative integer counts")
    result = ad.AnnData(X=counts, obs=obs, var=var)
    genes_by_species = {
        str(dataset.obs["species"].astype(str).iloc[0]): int(dataset.n_vars)
        for dataset in ordered
    }
    expressed_by_species = {
        str(dataset.obs["species"].astype(str).iloc[0]): int(
            np.asarray(sp.csr_matrix(dataset.X).getnnz(axis=0) > 0).sum()
        )
        for dataset in ordered
    }
    result.uns["eye_cross_species"] = {
        "created_at": datetime.now(UTC).isoformat(),
        "feature_space": "native_scprint_checkpoint_vocabularies",
        "species": sorted(result.obs["species"].astype(str).unique().tolist()),
        "cells_by_species": {
            str(key): int(value)
            for key, value in result.obs["species"].value_counts().items()
        },
        "genes_by_species": genes_by_species,
        "expressed_genes_by_species": expressed_by_species,
        "total_feature_columns": int(result.n_vars),
        "count_sum": int(result.X.sum()),
        "integer_counts": True,
        "feature_organism_ontology_term_ids": [
            value for value in organism_order if value in by_organism
        ],
        "scib_batch_key": "species",
        "scib_label_key": "celltype",
    }
    return result


def align_eye_to_reference_genes(
    source: ad.AnnData | Path | str,
    reference: ad.AnnData | Path | str,
    *,
    min_valid_genes_id: int = 500,
    min_nnz_genes: int = 200,
) -> ad.AnnData:
    """Align mouse symbols to a validated Ensembl feature table without a registry."""
    data = ad.read_h5ad(source) if isinstance(source, (Path, str)) else source.copy()
    reference_data = (
        ad.read_h5ad(reference, backed="r")
        if isinstance(reference, (Path, str))
        else reference
    )
    required_reference = {"symbol", "ensembl_gene_id", "organism"}
    missing_reference = sorted(required_reference - set(reference_data.var.columns))
    if missing_reference:
        raise KeyError(f"Reference gene table lacks columns: {missing_reference}")
    if not reference_data.var_names.is_unique:
        raise ValueError("Reference Ensembl gene identifiers are not unique")

    symbols = reference_data.var["symbol"].astype(str)
    symbol_counts = symbols.value_counts()
    unique_symbols = set(symbol_counts[symbol_counts == 1].index) - {"", "nan"}
    symbol_to_position = pd.Series(
        np.arange(reference_data.n_vars, dtype=np.int64), index=symbols
    ).loc[sorted(unique_symbols)]

    source_symbols = data.var["symbol"].astype(str)
    mapped_mask = source_symbols.isin(symbol_to_position.index).to_numpy()
    if not mapped_mask.any():
        raise ValueError("No eye mouse symbols map uniquely to the reference gene table")
    mapped_symbols = source_symbols.loc[mapped_mask]
    if len(mapped_symbols) < min_valid_genes_id:
        raise ValueError(
            "Eye dataset has fewer uniquely mapped genes than min_valid_genes_id: "
            f"{len(mapped_symbols)} < {min_valid_genes_id}"
        )
    target_positions = symbol_to_position.loc[mapped_symbols].to_numpy(dtype=np.int64)
    if len(np.unique(target_positions)) != len(target_positions):
        raise RuntimeError("Reference alignment produced duplicate Ensembl targets")

    source_counts = data.X
    if not sp.issparse(source_counts):
        source_counts = sp.csr_matrix(source_counts)
    keep_cells = np.asarray(source_counts.getnnz(axis=1) >= min_nnz_genes).ravel()
    if not keep_cells.any():
        raise ValueError("No eye cells pass the minimum non-zero expressed-gene threshold")

    mapped_counts = source_counts[:, mapped_mask]
    if not sp.issparse(mapped_counts):
        mapped_counts = sp.csr_matrix(mapped_counts)
    mapped_counts = mapped_counts.tocoo()
    aligned = sp.csr_matrix(
        (
            mapped_counts.data,
            (mapped_counts.row, target_positions[mapped_counts.col]),
        ),
        shape=(data.n_obs, reference_data.n_vars),
        dtype=np.int32,
    )
    aligned = aligned[keep_cells].tocsr()
    obs = data.obs.loc[keep_cells].copy()
    obs["nnz"] = aligned.getnnz(axis=1).astype(np.int32)
    obs["n_genes_by_counts"] = obs["nnz"].astype(np.int32)
    obs["total_counts"] = np.asarray(aligned.sum(axis=1)).ravel()
    result = ad.AnnData(
        X=aligned,
        obs=obs,
        var=reference_data.var.copy(),
        uns=data.uns.copy(),
    )
    if isinstance(reference_data, ad.AnnData) and reference_data.isbacked:
        reference_data.file.close()
    result.uns.setdefault("eye_cross_species", {})
    result.uns["eye_cross_species"]["scprint_preprocessing"] = {
        "input": {
            "cells": int(data.n_obs),
            "mouse_gene_symbols": int(data.n_vars),
            "count_sum": int(data.X.sum()),
        },
        "output": {
            "cells": int(result.n_obs),
            "checkpoint_ordered_ensembl_genes": int(result.n_vars),
            "count_sum": int(result.X.sum()),
        },
        "min_valid_genes_id": int(min_valid_genes_id),
        "min_nnz_genes": int(min_nnz_genes),
        "uniquely_mapped_input_symbols": int(mapped_mask.sum()),
        "ambiguous_reference_symbols_excluded": int((symbol_counts > 1).sum()),
        "reference_gene_table": (
            str(Path(reference).resolve())
            if isinstance(reference, (Path, str))
            else "in-memory AnnData"
        ),
        "registry_lookup": False,
        "pseudo_count_recovery": False,
    }
    return result


def preprocess_eye_dataset(
    source: ad.AnnData | Path | str,
    *,
    min_valid_genes_id: int = 500,
    min_nnz_genes: int = 200,
    reference: ad.AnnData | Path | str | None = None,
) -> ad.AnnData:
    """Run the same scPRINT-2 preprocessing contract on verified raw counts."""
    if reference is not None:
        return align_eye_to_reference_genes(
            source,
            reference,
            min_valid_genes_id=min_valid_genes_id,
            min_nnz_genes=min_nnz_genes,
        )
    from scdataloader import Preprocessor

    data = ad.read_h5ad(source) if isinstance(source, (Path, str)) else source.copy()
    if not sp.issparse(data.X):
        data.X = sp.csr_matrix(data.X)
    if (data.X.data < 0).any() or not np.equal(data.X.data, np.rint(data.X.data)).all():
        raise ValueError("scPRINT preprocessing requires non-negative integer counts")
    before = {
        "cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "count_sum": int(data.X.sum()),
    }
    result = Preprocessor(
        is_symbol=True,
        force_preprocess=True,
        skip_validate=True,
        do_postp=False,
        min_valid_genes_id=min_valid_genes_id,
        min_nnz_genes=min_nnz_genes,
    )(data)
    result.uns.setdefault("eye_cross_species", {})
    result.uns["eye_cross_species"]["scprint_preprocessing"] = {
        "input": before,
        "output": {"cells": int(result.n_obs), "genes": int(result.n_vars)},
        "min_valid_genes_id": int(min_valid_genes_id),
        "pseudo_count_recovery": False,
    }
    return result


def write_eye_manifest(data: ad.AnnData, output: Path | str) -> Path:
    """Write a compact audit manifest for a prepared benchmark dataset."""
    path = Path(output)
    payload = {
        "shape": [int(data.n_obs), int(data.n_vars)],
        "obs_names_unique": bool(data.obs_names.is_unique),
        "var_names_unique": bool(data.var_names.is_unique),
        "species": {
            str(key): int(value)
            for key, value in data.obs["species"].value_counts().items()
        },
        "cell_types": {
            str(key): int(value)
            for key, value in data.obs["celltype"].value_counts().items()
        },
        "count_sum": int(data.X.sum()),
        "integer_counts": bool(
            sp.issparse(data.X)
            and np.equal(data.X.data, np.rint(data.X.data)).all()
            and (data.X.data >= 0).all()
        ),
        "provenance": data.uns.get("eye_cross_species", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value.tolist()
            if isinstance(value, np.ndarray)
            else str(value),
        )
        + "\n"
    )
    return path


def write_eye_orthology_audit(
    inputs: Sequence[Path | str], output: Path | str
) -> Path:
    """Write one provenance row per species from prepared source H5ADs."""
    rows = []
    for source in inputs:
        data = ad.read_h5ad(source, backed="r")
        species_values = data.obs["species"].astype(str).unique().tolist()
        if len(species_values) != 1:
            data.file.close()
            raise ValueError(f"Expected one species in {source}, got {species_values}")
        geo = dict(data.uns.get("eye_geo_import", {}))
        orthology = dict(data.uns.get("eye_orthology", {}))
        rows.append(
            {
                "species": species_values[0],
                "cells": int(data.n_obs),
                "source_gene_rows": int(geo.get("source_gene_rows", data.n_vars)),
                "orthology_mode": str(orthology.get("mode", "strict_one_to_one")),
                "mapping_input_rows": int(orthology.get("input_rows", data.n_vars)),
                "mapping_unique_pairs": int(
                    orthology.get("unique_pairs", data.n_vars)
                ),
                "conflicting_source_genes": int(
                    orthology.get("conflicting_source_genes", 0)
                ),
                "conflicting_mouse_genes": int(
                    orthology.get("conflicting_mouse_genes", 0)
                ),
                "conflicting_target_genes": int(
                    orthology.get(
                        "conflicting_target_genes",
                        orthology.get("conflicting_mouse_genes", 0),
                    )
                ),
                "retained_mapping_pairs": int(
                    orthology.get("retained_pairs", data.n_vars)
                ),
                "retained_target_genes": int(
                    orthology.get("retained_targets", data.n_vars)
                ),
                "retained_one2one_pairs": int(
                    orthology.get("retained_one2one", data.n_vars)
                ),
                "retained_non_one2one_pairs": int(
                    orthology.get("retained_non_one2one", 0)
                ),
                "mapped_gene_rows_in_geo": int(
                    geo.get("mapped_gene_rows", data.n_vars)
                ),
                "mapped_source_gene_rows_in_geo": int(
                    geo.get("mapped_source_gene_rows", data.n_vars)
                ),
                "aggregated_target_gene_rows_in_geo": int(
                    geo.get("aggregated_target_gene_rows", 0)
                ),
                "integer_counts": bool(geo.get("integer_counts", False)),
                "source_sha256": str(geo.get("source_sha256", "")),
            }
        )
        data.file.close()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("species").to_csv(path, sep="\t", index=False)
    return path


def write_eye_native_gene_audit(
    inputs: Sequence[Path | str], output: Path | str
) -> Path:
    """Write the native input/checkpoint gene audit for each eye species."""
    rows = []
    for source in inputs:
        data = ad.read_h5ad(source, backed="r")
        species_values = data.obs["species"].astype(str).unique().tolist()
        source_organisms = (
            data.obs["source_organism_ontology_term_id"].astype(str).unique().tolist()
        )
        feature_organisms = (
            data.obs["organism_ontology_term_id"].astype(str).unique().tolist()
        )
        if not all(
            len(values) == 1
            for values in (species_values, source_organisms, feature_organisms)
        ):
            data.file.close()
            raise ValueError(f"Expected one species and organism in {source}")
        geo = dict(data.uns.get("eye_geo_import", {}))
        native = dict(data.uns.get("eye_native_checkpoint_alignment", {}))
        matrix = None
        if "expressed_checkpoint_genes" not in native or "count_sum" not in native:
            matrix = sp.csr_matrix(data.X[:])
        expressed_checkpoint_genes = int(
            native["expressed_checkpoint_genes"]
            if "expressed_checkpoint_genes" in native
            else np.asarray(matrix.getnnz(axis=0) > 0).sum()
        )
        count_sum = int(
            native["count_sum"]
            if "count_sum" in native
            else matrix.sum()
        )
        rows.append(
            {
                "species": species_values[0],
                "source_organism_ontology_term_id": source_organisms[0],
                "feature_organism_ontology_term_id": feature_organisms[0],
                "uses_supported_species_proxy": source_organisms[0]
                != feature_organisms[0],
                "source_gene_rows": int(geo.get("source_gene_rows", 0)),
                "checkpoint_genes": int(native.get("checkpoint_genes", data.n_vars)),
                "preprocessed_gene_overlap": int(
                    native.get("preprocessed_gene_overlap", data.n_vars)
                ),
                "expressed_checkpoint_genes": expressed_checkpoint_genes,
                "count_sum": count_sum,
                "integer_counts": bool(native.get("integer_counts", True)),
                "source_sha256": str(geo.get("source_sha256", "")),
            }
        )
        data.file.close()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("species").to_csv(path, sep="\t", index=False)
    return path
