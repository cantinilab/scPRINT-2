#!/usr/bin/env python3
"""Prepare Pandora lung-atlas exports for cross-species scPRINT embedding."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from scprint2.preprocessing.pandora import (
    BIOMART_URL,
    DEFAULT_FILTERED_LABELS,
    TARGET_FEATURE_SPACES,
    assemble_csc_h5ad,
    assemble_h5ad,
    combine_h5ads,
    fetch_mouse_gene_mapping,
    fetch_one_to_one_orthologs,
    filter_h5ad_by_obs_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapping = subparsers.add_parser("download-mapping")
    mapping.add_argument("--output", type=Path, required=True)
    mapping.add_argument("--url", default=BIOMART_URL)
    mapping.add_argument(
        "--target-feature-space",
        choices=sorted(TARGET_FEATURE_SPACES),
        default="human_one2one",
    )

    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--matrix", type=Path, required=True)
    assemble.add_argument("--obs", type=Path, required=True)
    assemble.add_argument("--var", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--organism-id", required=True)
    assemble.add_argument("--species", required=True)
    assemble.add_argument("--source-count-sum", type=int)
    assemble.add_argument("--source-feature-count", type=int)
    assemble.add_argument("--mapped-source-feature-count", type=int)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    combine.add_argument("--join", choices=("inner", "outer"), default="inner")

    filter_labels = subparsers.add_parser("filter-labels")
    filter_labels.add_argument("--input", type=Path, required=True)
    filter_labels.add_argument("--output", type=Path, required=True)
    filter_labels.add_argument("--label-key", default="cell_type_ontology_term_id")
    filter_labels.add_argument(
        "--exclude-label",
        action="append",
        default=list(DEFAULT_FILTERED_LABELS),
        help="Case-insensitive obs label to remove; repeat for multiple labels.",
    )
    filter_labels.add_argument("--expected-cells", type=int)

    csc = subparsers.add_parser("assemble-csc")
    csc.add_argument("--data", type=Path, required=True)
    csc.add_argument("--indices", type=Path, required=True)
    csc.add_argument("--indptr", type=Path, required=True)
    csc.add_argument("--genes", type=Path, required=True)
    csc.add_argument("--obs", type=Path, required=True)
    csc.add_argument("--orthologs", type=Path, required=True)
    csc.add_argument("--output", type=Path, required=True)
    csc.add_argument("--organism-id", required=True)
    csc.add_argument("--species", required=True)
    csc.add_argument(
        "--target-feature-space",
        choices=sorted(TARGET_FEATURE_SPACES),
        default="human_one2one",
    )

    args = parser.parse_args()
    if args.command == "download-mapping":
        is_mouse = args.target_feature_space == "mouse_homolog"
        table = (
            fetch_mouse_gene_mapping(args.url)
            if is_mouse
            else fetch_one_to_one_orthologs(args.url)
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
        metadata_path = args.output.with_suffix(".metadata.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "source": args.url,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                    "source_dataset": "mmusculus_gene_ensembl",
                    "target_species": "Mus musculus" if is_mouse else "Homo sapiens",
                    "target_feature_space": args.target_feature_space,
                    "filters": (
                        {"ambiguous_mouse_symbols": "excluded"}
                        if is_mouse
                        else {
                            "human_homology_type": "ortholog_one2one",
                            "human_orthology_confidence": "1",
                            "ambiguous_mouse_symbols": "excluded",
                        }
                    ),
                    "rows": len(table),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {len(table)} gene mappings to {args.output}")
    elif args.command == "assemble":
        result = assemble_h5ad(
            args.matrix,
            args.obs,
            args.var,
            args.output,
            args.organism_id,
            args.species,
            args.source_count_sum,
            args.source_feature_count,
            args.mapped_source_feature_count,
        )
        print(result)
    elif args.command == "combine":
        result = combine_h5ads(args.input, args.output, join=args.join)
        print(result)
    elif args.command == "filter-labels":
        result = filter_h5ad_by_obs_labels(
            args.input,
            args.output,
            label_key=args.label_key,
            excluded_labels=tuple(args.exclude_label),
            expected_cells=args.expected_cells,
        )
        print(result)
    else:
        result = assemble_csc_h5ad(
            args.data,
            args.indices,
            args.indptr,
            args.genes,
            args.obs,
            args.orthologs,
            args.output,
            args.organism_id,
            args.species,
            args.target_feature_space,
        )
        print(result)


if __name__ == "__main__":
    main()
