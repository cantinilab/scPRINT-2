#!/usr/bin/env python3
"""Benchmark PCA and scPRINT embeddings on the mapped Pandora lung atlas."""

from __future__ import annotations

import argparse
from pathlib import Path

from scprint2.preprocessing.pandora import (
    PandoraBenchmarkConfig,
    run_pandora_benchmark,
)


def _parse_embedding_spec(spec: str) -> tuple[str, Path, str]:
    try:
        name, source = spec.split("=", 1)
        path, key = source.rsplit(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "embedding spec must be NAME=PATH:OBSM_KEY"
        ) from exc
    if not name or not path or not key:
        raise argparse.ArgumentTypeError("embedding spec fields cannot be empty")
    return name, Path(path), key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-key", default="scprint_emb")
    parser.add_argument("--embedding-name")
    parser.add_argument(
        "--additional-embedding",
        action="append",
        default=[],
        type=_parse_embedding_spec,
        metavar="NAME=PATH:OBSM_KEY",
    )
    parser.add_argument("--batch-key", default="species")
    parser.add_argument("--label-key", default="cell_type_ontology_term_id")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=["unknown", "unmapped", "mix"],
        help="Case-insensitive label to exclude; repeat for multiple labels.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Add a deterministic uniform-random baseline matching X_pca shape.",
    )
    args = parser.parse_args()

    run_pandora_benchmark(
        PandoraBenchmarkConfig(
            expression=args.expression,
            embedding=args.embedding,
            output=args.output,
            embedding_key=args.embedding_key,
            embedding_name=args.embedding_name,
            additional_embeddings=tuple(args.additional_embedding),
            batch_key=args.batch_key,
            label_key=args.label_key,
            exclude_labels=tuple(args.exclude_label),
            seed=args.seed,
            n_jobs=args.n_jobs,
            include_random=args.include_random,
        )
    )


if __name__ == "__main__":
    main()
