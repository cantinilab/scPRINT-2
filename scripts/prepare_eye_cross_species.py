#!/usr/bin/env python3
"""Prepare the four-species eye benchmark from GEO integer count matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import pandas as pd

from scprint2.preprocessing.eye import (
    align_eye_to_checkpoint_genes,
    checkpoint_gene_vocabularies,
    combine_eye_native_species,
    combine_eye_ortholog_species,
    combine_eye_species,
    combine_eye_target_species,
    direct_symbol_to_ensembl_mapping,
    expand_observed_gene_aliases,
    extract_eye_annotations,
    load_geo_count_subset,
    merge_preferred_target_mappings,
    preprocess_eye_dataset,
    resolve_orthology_conflicts,
    resolve_source_orthology_targets,
    strict_one_to_one_mapping,
    update_eye_annotation_contract,
    write_eye_manifest,
    write_eye_native_gene_audit,
    write_eye_orthology_audit,
)


def _refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-annotations")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--species", required=True)
    export.add_argument("--source-taxon", required=True)
    export.add_argument("--feature-taxon")
    export.add_argument("--output", type=Path, required=True)

    rewrite = subparsers.add_parser("rewrite-annotations")
    rewrite.add_argument("--input", type=Path, required=True)
    rewrite.add_argument("--species", required=True)
    rewrite.add_argument("--source-taxon", required=True)
    rewrite.add_argument("--feature-taxon")
    rewrite.add_argument("--output", type=Path, required=True)

    build = subparsers.add_parser("build-species")
    build.add_argument("--counts", type=Path, required=True)
    build.add_argument("--annotations", type=Path, required=True)
    build.add_argument("--mapping", type=Path, action="append", default=[])
    build.add_argument("--source-column", action="append", default=[])
    build.add_argument("--target-column", default="mouse_gene")
    build.add_argument("--identity-mapping", action="store_true")
    build.add_argument("--resolve-orthology-conflicts", action="store_true")
    build.add_argument(
        "--mapping-mode",
        choices=(
            "strict-one-to-one",
            "target-unique-orthology",
            "direct-human-symbol",
            "direct-human-symbol-with-orthology-fallback",
            "source-unique-target-aggregated-orthology",
        ),
    )
    build.add_argument("--normalize-terminal-np-suffixes", action="store_true")
    build.add_argument(
        "--direct-registry",
        type=Path,
        help="Human symbol/ENSG registry used by the hybrid mapping mode.",
    )
    build.add_argument("--direct-symbol-column", default="source_gene_symbol")
    build.add_argument("--direct-ensembl-column", default="source_ensembl_gene_id")
    build.add_argument(
        "--target-id-type", choices=("symbol", "ensembl"), default="symbol"
    )
    build.add_argument("--chunksize", type=int, default=64)
    build.add_argument("--output", type=Path, required=True)

    combine = subparsers.add_parser("combine")
    combine.add_argument("--input", type=Path, action="append", required=True)
    combine.add_argument("--min-shared-genes", type=int, default=1000)
    combine.add_argument("--output", type=Path, required=True)

    align_checkpoint = subparsers.add_parser("align-checkpoint")
    align_checkpoint.add_argument("--input", type=Path, required=True)
    align_checkpoint.add_argument("--checkpoint", type=Path, required=True)
    align_checkpoint.add_argument("--min-checkpoint-genes", type=int, default=15_000)
    align_checkpoint.add_argument("--min-input-gene-overlap", type=int, default=15_000)
    align_checkpoint.add_argument("--output", type=Path, required=True)

    combine_native = subparsers.add_parser("combine-native")
    combine_native.add_argument("--input", type=Path, action="append", required=True)
    combine_native.add_argument("--checkpoint", type=Path, required=True)
    combine_native.add_argument("--min-genes-per-species", type=int, default=15_000)
    combine_native.add_argument("--output", type=Path, required=True)

    combine_ortholog = subparsers.add_parser("combine-ortholog-union")
    combine_ortholog.add_argument("--input", type=Path, action="append", required=True)
    combine_ortholog.add_argument("--min-genes-per-species", type=int, default=15_000)
    combine_ortholog.add_argument("--min-common-genes", type=int, default=10_000)
    combine_ortholog.add_argument("--output", type=Path, required=True)

    combine_target = subparsers.add_parser("combine-target-union")
    combine_target.add_argument("--input", type=Path, action="append", required=True)
    combine_target.add_argument("--feature-taxon", required=True)
    combine_target.add_argument("--feature-space", required=True)
    combine_target.add_argument("--min-genes-per-species", type=int, default=14_000)
    combine_target.add_argument("--output", type=Path, required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.add_argument("--input", type=Path, required=True)
    preprocess.add_argument("--min-valid-genes", type=int, default=500)
    preprocess.add_argument("--min-nnz-genes", type=int, default=200)
    preprocess.add_argument(
        "--reference",
        type=Path,
        help="Validated checkpoint-ordered mouse gene table; avoids registry lookup.",
    )
    preprocess.add_argument("--output", type=Path, required=True)
    preprocess.add_argument("--manifest", type=Path, required=True)
    preprocess.add_argument("--complete", type=Path, required=True)

    manifest = subparsers.add_parser("write-manifest")
    manifest.add_argument("--input", type=Path, required=True)
    manifest.add_argument("--manifest", type=Path, required=True)
    manifest.add_argument("--complete", type=Path, required=True)

    orthology = subparsers.add_parser("write-orthology-audit")
    orthology.add_argument("--input", type=Path, action="append", required=True)
    orthology.add_argument("--output", type=Path, required=True)

    native_audit = subparsers.add_parser("write-native-gene-audit")
    native_audit.add_argument("--input", type=Path, action="append", required=True)
    native_audit.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export-annotations":
        _refuse_overwrite(args.output)
        source = ad.read_h5ad(args.input, backed="r")
        annotations = extract_eye_annotations(
            source,
            species=args.species,
            source_organism_ontology_term_id=args.source_taxon,
            feature_organism_ontology_term_id=args.feature_taxon,
        )
        annotations.to_csv(args.output, sep="\t")
        source.file.close()
        print(f"Wrote {len(annotations)} expert annotations to {args.output}")
        return


    if args.command == "rewrite-annotations":
        _refuse_overwrite(args.output)
        annotations = pd.read_csv(
            args.input,
            sep="\t",
            index_col="source_obs_name",
            dtype=str,
            keep_default_na=False,
        )
        result = update_eye_annotation_contract(
            annotations,
            species=args.species,
            source_organism_ontology_term_id=args.source_taxon,
            feature_organism_ontology_term_id=args.feature_taxon,
        )
        result.to_csv(args.output, sep="\t")
        print(f"Wrote {len(result)} updated annotations to {args.output}")
        return

    if args.command == "build-species":
        _refuse_overwrite(args.output)
        annotations = pd.read_csv(
            args.annotations,
            sep="\t",
            index_col="source_obs_name",
            dtype=str,
            keep_default_na=False,
        )
        if args.identity_mapping:
            if args.mapping or args.source_column:
                raise ValueError("Identity mapping cannot be combined with mapping files")
            mapping = None
            mapping_audit = {"mode": "mouse_symbol_identity"}
        else:
            if not args.mapping or not args.source_column:
                raise ValueError(
                    "Mapped species require --mapping and --source-column"
                )
            frames = [
                pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
                for path in args.mapping
            ]
            gene_column = pd.read_csv(args.counts, nrows=0).columns[0]
            observed_genes = pd.read_csv(
                args.counts,
                usecols=[gene_column],
            )[gene_column].astype(str)
            mapping_mode = args.mapping_mode
            if mapping_mode is None:
                mapping_mode = (
                    "target-unique-orthology"
                    if args.resolve_orthology_conflicts
                    else "strict-one-to-one"
                )
            if mapping_mode == "direct-human-symbol":
                if len(args.source_column) != 1:
                    raise ValueError(
                        "Direct human symbol mapping requires one --source-column"
                    )
                mapping, mapping_audit = direct_symbol_to_ensembl_mapping(
                    pd.concat(frames, ignore_index=True),
                    observed_genes=observed_genes,
                    symbol_col=args.source_column[0],
                    ensembl_col=args.target_column,
                    normalize_terminal_np_suffixes=(
                        args.normalize_terminal_np_suffixes
                    ),
                )
            elif mapping_mode == "direct-human-symbol-with-orthology-fallback":
                if args.direct_registry is None:
                    raise ValueError("Hybrid mapping requires --direct-registry")
                direct_registry = pd.read_csv(
                    args.direct_registry,
                    sep="\t" if args.direct_registry.suffix == ".tsv" else ",",
                )
                direct, direct_audit = direct_symbol_to_ensembl_mapping(
                    direct_registry,
                    observed_genes=observed_genes,
                    symbol_col=args.direct_symbol_column,
                    ensembl_col=args.direct_ensembl_column,
                    normalize_terminal_np_suffixes=(
                        args.normalize_terminal_np_suffixes
                    ),
                )
                frames = expand_observed_gene_aliases(
                    frames,
                    source_columns=args.source_column,
                    observed_genes=observed_genes,
                )
                fallback, fallback_audit = resolve_source_orthology_targets(
                    frames,
                    source_col="_observed_source_gene",
                    target_col=args.target_column,
                )
                mapping, mapping_audit = merge_preferred_target_mappings(
                    direct, fallback
                )
                mapping_audit["direct"] = direct_audit
                mapping_audit["orthology_fallback"] = fallback_audit
            else:
                frames = expand_observed_gene_aliases(
                    frames,
                    source_columns=args.source_column,
                    observed_genes=observed_genes,
                )
                if mapping_mode == "source-unique-target-aggregated-orthology":
                    mapping_function = resolve_source_orthology_targets
                elif mapping_mode == "target-unique-orthology":
                    mapping_function = resolve_orthology_conflicts
                else:
                    mapping_function = strict_one_to_one_mapping
                mapping, mapping_audit = mapping_function(
                    frames,
                    source_col="_observed_source_gene",
                    target_col=args.target_column,
                )
        result = load_geo_count_subset(
            args.counts,
            annotations,
            mapping,
            chunksize=args.chunksize,
            aggregate_duplicate_targets=(
                not args.identity_mapping
                and mapping_mode
                in {
                    "direct-human-symbol",
                    "direct-human-symbol-with-orthology-fallback",
                    "source-unique-target-aggregated-orthology",
                }
            ),
            target_id_type=args.target_id_type,
        )
        result.uns["eye_orthology"] = mapping_audit
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        print(json.dumps(mapping_audit, indent=2, sort_keys=True))
        return

    if args.command == "combine":
        _refuse_overwrite(args.output)
        sources = [ad.read_h5ad(path) for path in args.input]
        result = combine_eye_species(
            sources, min_shared_genes=args.min_shared_genes
        )
        result.uns["eye_cross_species"]["source_h5ads"] = [
            str(path.resolve()) for path in args.input
        ]
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        return

    if args.command == "align-checkpoint":
        _refuse_overwrite(args.output)
        vocabularies, _ = checkpoint_gene_vocabularies(args.checkpoint)
        result = align_eye_to_checkpoint_genes(
            args.input,
            vocabularies,
            min_checkpoint_genes=args.min_checkpoint_genes,
            min_input_gene_overlap=args.min_input_gene_overlap,
        )
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        print(json.dumps(result.uns["eye_native_checkpoint_alignment"], indent=2))
        return

    if args.command == "combine-native":
        _refuse_overwrite(args.output)
        _, organism_order = checkpoint_gene_vocabularies(args.checkpoint)
        sources = [ad.read_h5ad(path) for path in args.input]
        result = combine_eye_native_species(
            sources,
            organism_order=organism_order,
            min_genes_per_species=args.min_genes_per_species,
        )
        result.uns["eye_cross_species"]["source_h5ads"] = [
            str(path.resolve()) for path in args.input
        ]
        result.uns["eye_cross_species"]["checkpoint"] = str(
            args.checkpoint.resolve()
        )
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        print(json.dumps(result.uns["eye_cross_species"], indent=2))
        return

    if args.command == "combine-ortholog-union":
        _refuse_overwrite(args.output)
        sources = [ad.read_h5ad(path) for path in args.input]
        result = combine_eye_ortholog_species(
            sources,
            min_genes_per_species=args.min_genes_per_species,
            min_common_genes=args.min_common_genes,
        )
        result.uns["eye_cross_species"]["source_h5ads"] = [
            str(path.resolve()) for path in args.input
        ]
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        print(json.dumps(result.uns["eye_cross_species"], indent=2))
        return

    if args.command == "combine-target-union":
        _refuse_overwrite(args.output)
        sources = [ad.read_h5ad(path) for path in args.input]
        result = combine_eye_target_species(
            sources,
            target_organism_ontology_term_id=args.feature_taxon,
            feature_space=args.feature_space,
            min_genes_per_species=args.min_genes_per_species,
        )
        result.uns["eye_cross_species"]["source_h5ads"] = [
            str(path.resolve()) for path in args.input
        ]
        result.write_h5ad(args.output, compression="gzip")
        print(result)
        print(json.dumps(result.uns["eye_cross_species"], indent=2))
        return

    if args.command == "write-manifest":
        for path in (args.manifest, args.complete):
            _refuse_overwrite(path)
        result = ad.read_h5ad(args.input)
        write_eye_manifest(result, args.manifest)
        args.complete.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(args.input.resolve()),
                    "manifest": str(args.manifest.resolve()),
                    "shape": [int(result.n_obs), int(result.n_vars)],
                },
                indent=2,
            )
            + "\n"
        )
        print(result)
        return

    if args.command == "write-orthology-audit":
        _refuse_overwrite(args.output)
        write_eye_orthology_audit(args.input, args.output)
        print(args.output)
        return

    if args.command == "write-native-gene-audit":
        _refuse_overwrite(args.output)
        write_eye_native_gene_audit(args.input, args.output)
        print(args.output)
        return

    for path in (args.output, args.manifest, args.complete):
        _refuse_overwrite(path)
    result = preprocess_eye_dataset(
        args.input,
        min_valid_genes_id=args.min_valid_genes,
        min_nnz_genes=args.min_nnz_genes,
        reference=args.reference,
    )
    result.write_h5ad(args.output, compression="gzip")
    write_eye_manifest(result, args.manifest)
    args.complete.write_text(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "manifest": str(args.manifest.resolve()),
                "shape": [int(result.n_obs), int(result.n_vars)],
            },
            indent=2,
        )
        + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
