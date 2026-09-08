#!/usr/bin/env python3
"""Fetch high-confidence source-species orthologs into human ENSG space."""

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
ORTHOLOG_PREFIXES = {
    "macaque_fascicularis": "mfascicularis",
    "mouse": "mmusculus",
    "pig": "sscrofa",
}
OUTPUT_COLUMNS = [
    "human_gene_symbol",
    "human_ensembl_gene_id",
    "source_gene_symbol",
    "source_ensembl_gene_id",
    "homology_type",
    "orthology_confidence",
    "source_homolog_percent_identity",
]


def human_target_biomart_query(prefix: str) -> str:
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
        '<Dataset name="hsapiens_gene_ensembl" interface="default">'
        f"{xml_attributes}</Dataset></Query>"
    )


def fetch_human_target_orthologs(
    species: str,
    *,
    url: str = BIOMART_URL,
    timeout: int = 180,
) -> tuple[pd.DataFrame, str]:
    prefix = ORTHOLOG_PREFIXES[species]
    query = human_target_biomart_query(prefix)
    response = requests.get(url, params={"query": query}, timeout=timeout)
    response.raise_for_status()
    table = pd.read_csv(
        io.BytesIO(response.content),
        sep="\t",
        header=None,
        names=OUTPUT_COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    if table.shape[1] != len(OUTPUT_COLUMNS):
        raise ValueError("Unexpected BioMart response width")
    if table.empty or table.iloc[:, 0].astype(str).str.startswith("Query ERROR").any():
        raise RuntimeError("BioMart human-target query failed")
    table = table.loc[
        table["human_ensembl_gene_id"].ne("")
        & table["source_ensembl_gene_id"].ne("")
        & table["homology_type"].str.startswith("ortholog_")
        & table["orthology_confidence"].eq("1")
    ].drop_duplicates()
    if table.empty:
        raise RuntimeError(f"No high-confidence human orthologs returned for {species}")
    return (
        table.sort_values(
            ["human_ensembl_gene_id", "source_ensembl_gene_id", "homology_type"]
        ).reset_index(drop=True),
        query,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", choices=sorted(ORTHOLOG_PREFIXES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default=BIOMART_URL)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = args.output.with_suffix(".metadata.json")
    if args.output.exists() or metadata.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output} or {metadata}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table, query = fetch_human_target_orthologs(
        args.species, url=args.url, timeout=args.timeout
    )
    table.to_csv(args.output, sep="\t", index=False)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "species": args.species,
        "target_species": "human",
        "url": args.url,
        "query": query,
        "rows": int(len(table)),
        "source_gene_symbols": int(
            table.loc[table["source_gene_symbol"].ne(""), "source_gene_symbol"].nunique()
        ),
        "source_ensembl_gene_ids": int(table["source_ensembl_gene_id"].nunique()),
        "human_ensembl_gene_ids": int(table["human_ensembl_gene_id"].nunique()),
        "homology_types": {
            str(key): int(value)
            for key, value in table["homology_type"].value_counts().items()
        },
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
