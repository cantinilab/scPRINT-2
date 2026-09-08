#!/usr/bin/env python3
"""Score fresh task3 embeddings with scib 1.1.3 and precomputed graphs."""

from __future__ import annotations

import argparse
from pathlib import Path

from scprint2.evaluation.task3_scib import (
    BATCH_METRICS,
    BIO_METRICS,
    TASK3_SCIB113_EXPECTED_VERSIONS,
    load_aligned_embedding,
    mean_available,
    parse_embedding_spec,
    parse_obs_name_replace_spec,
    prepare_expression,
    run_scib113_precomputed_benchmark,
    safe_name,
    sha256,
    sparse_sha256,
    write_precomputed_graph,
)

EXPECTED_VERSIONS = TASK3_SCIB113_EXPECTED_VERSIONS

_sha256 = sha256
_sparse_sha256 = sparse_sha256
_prepare_expression = prepare_expression
_aligned_embedding = load_aligned_embedding
_safe_name = safe_name
_write_precomputed_graph = write_precomputed_graph
_mean = mean_available


def _parse_embedding(spec: str) -> tuple[str, Path, str]:
    try:
        return parse_embedding_spec(spec)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "embedding must be METHOD=PATH:OBSM_KEY"
        ) from error


def _parse_obs_name_replace(spec: str) -> tuple[str, str]:
    try:
        return parse_obs_name_replace_spec(spec)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "obs-name-replace must be SOURCE_SUFFIX=EMBEDDING_SUFFIX"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--embedding", action="append", required=True, type=_parse_embedding
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-key", default="orig.ident")
    parser.add_argument("--label-key", default="cell_type_ontology_term_id")
    parser.add_argument("--n-cores", type=int, default=8)
    parser.add_argument("--obs-name-replace", type=_parse_obs_name_replace)
    return parser.parse_args()


def main() -> None:
    run_scib113_precomputed_benchmark(parse_args())


if __name__ == "__main__":
    main()
