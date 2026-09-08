#!/usr/bin/env python3
"""Fetch full high-confidence mouse ortholog tables for the eye benchmark."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BIOMART_URL = "https://useast.ensembl.org/biomart/martservice"
SPECIES_CONFIG = {
    "human": {
        "native_dataset": "hsapiens_gene_ensembl",
        "ortholog_prefix": "hsapiens",
    },
    "macaque_fascicularis": {
        # small-v2 has rhesus but not cynomolgus. Gene symbols are resolved
        # against the supported rhesus vocabulary while source provenance stays
        # NCBITaxon:9541.
        "native_dataset": "mmulatta_gene_ensembl",
        # TranscriptFormer/PCA start from the actual GEO source species, not
        # from the rhesus proxy used only for scPRINT-2 tokenization.
        "ortholog_prefix": "mfascicularis",
    },
    "mouse": {
        "native_dataset": "mmusculus_gene_ensembl",
        "ortholog_prefix": None,
    },
    "pig": {
        "native_dataset": "sscrofa_gene_ensembl",
        "ortholog_prefix": "sscrofa",
    },
}
OUTPUT_COLUMNS = [
    "mouse_gene_symbol",
    "mouse_ensembl_gene_id",
    "source_gene_symbol",
    "source_ensembl_gene_id",
    "homology_type",
    "orthology_confidence",
    "source_homolog_percent_identity",
]


def biomart_query(prefix: str) -> str:
    attributes = (
        "external_gene_name",
        "ensembl_gene_id",
        f"{prefix}_homolog_associated_gene_name",
        f"{prefix}_homolog_ensembl_gene",
        f"{prefix}_homolog_orthology_type",
        f"{prefix}_homolog_orthology_confidence",
        f"{prefix}_homolog_perc_id",
    )
    xml_attributes = "".join(f'<Attribute name="{name}"/>' for name in attributes)
    return (
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" datasetConfigVersion="0.6">'
        '<Dataset name="mmusculus_gene_ensembl" interface="default">'
        f"{xml_attributes}</Dataset></Query>"
    )


def native_gene_query(dataset: str) -> str:
    return (
        '<Query virtualSchemaName="default" formatter="TSV" header="0" '
        'uniqueRows="1" datasetConfigVersion="0.6">'
        f'<Dataset name="{dataset}" interface="default">'
        '<Attribute name="external_gene_name"/>'
        '<Attribute name="ensembl_gene_id"/>'
        "</Dataset></Query>"
    )


def _request_table(
    query: str,
    columns: list[str],
    *,
    url: str,
    timeout: int,
) -> pd.DataFrame:
    response = requests.get(url, params={"query": query}, timeout=timeout)
    response.raise_for_status()
    table = pd.read_csv(
        io.BytesIO(response.content),
        sep="\t",
        header=None,
        names=columns,
        dtype=str,
        keep_default_na=False,
    )
    if table.shape[1] != len(columns):
        raise ValueError("Unexpected BioMart response width")
    if table.empty or table.iloc[:, 0].astype(str).str.startswith("Query ERROR").any():
        raise RuntimeError("BioMart query failed")
    return table


def fetch_native_genes(
    species: str,
    *,
    url: str = BIOMART_URL,
    timeout: int = 180,
) -> pd.DataFrame:
    dataset = SPECIES_CONFIG[species]["native_dataset"]
    table = _request_table(
        native_gene_query(str(dataset)),
        ["source_gene_symbol", "source_ensembl_gene_id"],
        url=url,
        timeout=timeout,
    )
    table = table.loc[
        table["source_gene_symbol"].ne("")
        & table["source_ensembl_gene_id"].ne("")
    ].drop_duplicates()
    if table.empty:
        raise RuntimeError(f"No native genes returned for {species}")
    return table.sort_values(
        ["source_gene_symbol", "source_ensembl_gene_id"]
    ).reset_index(drop=True)


def fetch_species_orthologs(
    species: str,
    *,
    url: str = BIOMART_URL,
    timeout: int = 180,
) -> pd.DataFrame:
    prefix = SPECIES_CONFIG[species]["ortholog_prefix"]
    if prefix is None:
        raise ValueError(f"No mouse ortholog query is needed for {species}")
    table = _request_table(
        biomart_query(str(prefix)),
        OUTPUT_COLUMNS,
        url=url,
        timeout=timeout,
    )
    table = table.loc[
        table["mouse_gene_symbol"].ne("")
        & table["source_gene_symbol"].ne("")
        & table["homology_type"].str.startswith("ortholog_")
        & table["orthology_confidence"].eq("1")
    ].drop_duplicates()
    if table.empty:
        raise RuntimeError(f"No high-confidence orthologs returned for {species}")
    return table.sort_values(
        ["mouse_gene_symbol", "source_gene_symbol", "homology_type"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=BIOMART_URL)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for species, config in SPECIES_CONFIG.items():
        native_output = args.output_dir / f"{species}_native_genes_biomart.tsv"
        native_metadata = native_output.with_suffix(".metadata.json")
        if native_output.exists() or native_metadata.exists():
            raise FileExistsError(
                f"Refusing to overwrite {native_output} or {native_metadata}"
            )
        native = fetch_native_genes(
            species,
            url=args.url,
            timeout=args.timeout,
        )
        native.to_csv(native_output, sep="\t", index=False)
        native_digest = hashlib.sha256(native_output.read_bytes()).hexdigest()
        native_payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "species": species,
            "feature_species_proxy": "macaque_rhesus"
            if species == "macaque_fascicularis"
            else None,
            "url": args.url,
            "query": native_gene_query(str(config["native_dataset"])),
            "rows": int(len(native)),
            "source_gene_symbols": int(native["source_gene_symbol"].nunique()),
            "source_ensembl_gene_ids": int(
                native["source_ensembl_gene_id"].nunique()
            ),
            "sha256": native_digest,
        }
        native_metadata.write_text(
            json.dumps(native_payload, indent=2, sort_keys=True) + "\n"
        )
        payload = {"native": native_payload}
        if config["ortholog_prefix"] is None:
            summary[species] = payload
            print(species, json.dumps(payload, sort_keys=True))
            continue
        output = args.output_dir / f"{species}_to_mouse_biomart.tsv"
        metadata = output.with_suffix(".metadata.json")
        if output.exists() or metadata.exists():
            raise FileExistsError(f"Refusing to overwrite {output} or {metadata}")
        table = fetch_species_orthologs(
            species, url=args.url, timeout=args.timeout
        )
        table.to_csv(output, sep="\t", index=False)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        homology_types = {
            str(key): int(value)
            for key, value in table["homology_type"].value_counts().items()
        }
        ortholog_payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "species": species,
            "url": args.url,
            "query": biomart_query(str(config["ortholog_prefix"])),
            "rows": int(len(table)),
            "source_gene_symbols": int(table["source_gene_symbol"].nunique()),
            "mouse_gene_symbols": int(table["mouse_gene_symbol"].nunique()),
            "homology_types": homology_types,
            "sha256": digest,
        }
        metadata.write_text(
            json.dumps(ortholog_payload, indent=2, sort_keys=True) + "\n"
        )
        payload["orthologs"] = ortholog_payload
        summary[species] = payload
        print(species, json.dumps(payload, sort_keys=True))
    (args.output_dir / "BIOMART.COMPLETE").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
